import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go

# Funciones de evaluación
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run():
    contaminant = st.sidebar.selectbox("Selecciona un contaminante", ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5", "SO2"])
    
    # Cargar modelo específico
    model_path = f"results/{contaminant}_noroeste/model_{contaminant}_noroeste.pkl"
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
            st.success(f"Modelo para {contaminant} cargado exitosamente.")
    except FileNotFoundError:
        st.error(f"No se encontró el modelo para {contaminant} en Noroeste.")
        return

    # Cargar datos
    data_path = "df_20222024_definitivo.csv"
    try:
        df = pd.read_csv(data_path)
        st.success("Datos cargados exitosamente.")
    except FileNotFoundError:
        st.error("El archivo de datos no se encontró.")
        return

    # Preprocesamiento y predicción
    dates = pd.to_datetime(df['date'])
    df.drop('date', axis=1, inplace=True)
    columns_to_keep = model.feature_names_in_
    X = df[columns_to_keep]
    last_X = X.iloc[-120:]
    real_values = df[f"{contaminant}_noroeste"].iloc[-240:]
    predicted_values = model.predict(X.iloc[-240:])
    forecast = model.predict(last_X)
    forecast_dates = pd.date_range(start=dates.iloc[-1] + pd.Timedelta(hours=1), periods=120, freq='H')

    # Visualización
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates.iloc[-240:], y=real_values, mode='lines', name='Valores reales', line=dict(color='gray', width=2)))
    fig.add_trace(go.Scatter(x=dates.iloc[-240:], y=predicted_values, mode='lines', name='Predicciones', line=dict(color='purple', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red', width=2, dash='dot')))
    fig.update_layout(title=f"Pronóstico para {contaminant} en Noroeste", xaxis_title="Fecha", yaxis_title="Concentración", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # Métricas
    mse = mean_squared_error(real_values, predicted_values)
    mae = mean_absolute_error(real_values, predicted_values)
    r2 = r2_score(real_values, predicted_values)
    rmse_value = rmse(real_values, predicted_values)
    mape_value = mape(real_values, predicted_values)

    # Agregar datos de forecasting a la tabla
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast
    })

    # Mostrar la tabla de forecast en Streamlit
    st.dataframe(forecast_df, width=800, height=400, hide_index = True)

    # Crear cajas HTML para cada métrica
    metric_boxes = f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-top: 20px; flex-wrap: wrap;">
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; width: 120px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin: 0; font-size: 1rem; color: #155724;">MSE</h3>
            <p style="margin: 5px 0; font-size: 1.2rem; font-weight: bold; color: #155724;">{mse:.2f}</p>
        </div>
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; width: 120px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin: 0; font-size: 1rem; color: #155724;">MAE</h3>
            <p style="margin: 5px 0; font-size: 1.2rem; font-weight: bold; color: #155724;">{mae:.2f}</p>
        </div>
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; width: 120px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin: 0; font-size: 1rem; color: #155724;">R2 Score</h3>
            <p style="margin: 5px 0; font-size: 1.2rem; font-weight: bold; color: #155724;">{r2:.2f}</p>
        </div>
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; width: 120px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin: 0; font-size: 1rem; color: #155724;">RMSE</h3>
            <p style="margin: 5px 0; font-size: 1.2rem; font-weight: bold; color: #155724;">{rmse_value:.2f}</p>
        </div>
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; width: 120px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin: 0; font-size: 1rem; color: #155724;">MAPE</h3>
            <p style="margin: 5px 0; font-size: 1.2rem; font-weight: bold; color: #155724;">{mape_value:.2f} %</p>
        </div>
    </div>

    """

    # Mostrar las cajas HTML en Streamlit
    st.markdown(metric_boxes, unsafe_allow_html=True)
