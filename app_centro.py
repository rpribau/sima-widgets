import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

import streamlit as st

def run():
    st.write("Aquí va la lógica para la estación Norte")
    # Tu código de la estación "norte" aquí
    columnas = [
    "CO_centro", "NO_centro", "NO2_centro", "NOX_centro", "O3_centro", "PM10_centro", "PM2.5_centro", 
    "SO2_centro"
    ]

    # Variables para identificar si se seleccionó algún componente/estación
    componente_estacion = None

    # Mostrar los botones en el sidebar
    st.sidebar.title("Selecciona el Componente y Estación")

    for columna in columnas:
        if st.sidebar.button(columna):
            componente_estacion = columna

    # Verificar si se seleccionó un componente/estación
    if componente_estacion:
        st.write(f"Has seleccionado: {componente_estacion}")

        # Cargar los datos
        df = pd.read_csv('data/df_20222024_con_horas.csv')

        # Ordenar por fecha para garantizar que tomemos los últimos 120 valores en base a la fecha
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')

        # Asegurarte que la columna existe en el dataset
        if componente_estacion in df.columns:
            # Preparar los datos
            X = df.drop([componente_estacion, 'date'], axis=1)  # Quitamos la columna de fecha y el componente objetivo
            y = df[componente_estacion]

            # Dividir los datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Modelo Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)

            # Tomar los últimos 120 valores basados en la fecha
            df_test_last_120 = df.tail(120)
            X_test_last_120 = df_test_last_120.drop([componente_estacion, 'date'], axis=1)
            y_test_last_120 = df_test_last_120[componente_estacion]
            y_pred_last_120 = rf_model.predict(X_test_last_120)

            # Evaluar el modelo
            mse_rf = mean_squared_error(y_test_last_120, y_pred_last_120)
            r2_rf = r2_score(y_test_last_120, y_pred_last_120)
            st.write(f'Random Forest - MSE: {mse_rf}, R2: {r2_rf}')

            # Gráfico de barras y líneas para comparar predicciones y valores reales (últimos 120 datos)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_test_last_120['date'], y=y_test_last_120, name='Valores Reales'))
            fig.add_trace(go.Scatter(x=df_test_last_120['date'], y=y_pred_last_120, mode='lines', name='Predicciones'))

            fig.update_layout(title=f"Comparación de {componente_estacion} (Últimos 120 datos): Predicción vs Real",
                            xaxis_title='Fecha', yaxis_title='Concentración')

            st.plotly_chart(fig)

            # Crear tabla con date, reales, predicciones y error
            df_resultados = pd.DataFrame({
                'date': df_test_last_120['date'],
                'Valores Reales': y_test_last_120.values,
                'Predicciones': y_pred_last_120,
                'Error': (y_test_last_120.values - y_pred_last_120)  # Error = Real - Predicho
            })

            # Mostrar la tabla en la app
            st.write("Tabla de Resultados:")
            st.dataframe(df_resultados, width=1000, height=300)

        else:
            st.error(f"La columna {componente_estacion} no se encuentra en el dataset.")
    else:
        st.write("Selecciona un componente y estación en el sidebar.")



# Lista de columnas disponibles
