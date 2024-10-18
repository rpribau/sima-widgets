import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import numpy as np

def run():
    st.write("Aquí va la lógica para la estación Centro")

    columnas = [
        "CO_centro", "NO_centro", "NO2_centro", "NOX_centro", "O3_centro", "PM10_centro", "PM2.5_centro", 
        "SO2_centro"
    ]

    componente_estacion = None

    st.sidebar.title("Selecciona el Componente y Estación")

    for columna in columnas:
        if st.sidebar.button(columna):
            componente_estacion = columna

    if componente_estacion:
        st.write(f"Has seleccionado: {componente_estacion}")

        df = pd.read_csv('data/df_20222024_con_horas.csv')

        df = df.sort_values(by='date')

        if componente_estacion in df.columns:
            # Asegurarnos de dropear la columna que queremos predecir del dataset
            X = df.drop([componente_estacion, 'date'], axis=1)  # Quitamos la columna objetivo y 'date'
            y = df[componente_estacion]

            # Directorio donde guardaremos los modelos
            modelo_path = f'models/centro/{componente_estacion}.pkl'

            # Si el modelo ya existe, cargarlo; de lo contrario, entrenarlo y guardarlo
            if os.path.exists(modelo_path):
                with open(modelo_path, 'rb') as file:
                    best_rf = pickle.load(file)
                st.write(f"Modelo cargado desde {modelo_path}")
            else:
                # Dividir los datos en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Configurar el modelo y los hiperparámetros para GridSearchCV
                rf = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [100],
                    'max_depth': [None],
                    'min_samples_split': [5]
                }

                # Definir la validación cruzada con KFold
                kfold = KFold(n_splits=3, shuffle=True, random_state=42)

                # Realizar GridSearchCV con validación cruzada
                grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Obtener el mejor estimador
                best_rf = grid_search.best_estimator_

                # Guardar el modelo entrenado
                os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
                with open(modelo_path, 'wb') as file:
                    pickle.dump(best_rf, file)
                st.write(f"Modelo entrenado con validación cruzada y guardado en {modelo_path}")

            # Predicciones sobre los últimos 120 valores conocidos
            df_test_last_120 = df.tail(120)
            X_test_last_120 = df_test_last_120.drop([componente_estacion, 'date'], axis=1)
            y_test_last_120 = df_test_last_120[componente_estacion]
            y_pred_last_120 = best_rf.predict(X_test_last_120)

            # Generar fechas futuras para los próximos 5 días (120 horas)
            last_date = pd.to_datetime(df['date'].max())  # Obtener la última fecha del dataset
            future_dates = pd.date_range(last_date, periods=121, freq='H')[1:]  # Generar 120 horas futuras

            # Generar datos futuros basados en las diferencias de las últimas 120 horas
            X_differences = X_test_last_120.diff().mean()  # Obtener la media de las diferencias de las últimas horas

            X_nuevos_120 = X_test_last_120.tail(1).copy()  # Copiar la última fila de X
            X_nuevos_futuros = [X_nuevos_120]

            for i in range(120):
                X_nuevo = X_nuevos_120 + X_differences  # Agregar las diferencias a la fila anterior
                X_nuevos_futuros.append(X_nuevo)
                X_nuevos_120 = X_nuevo.copy()

            X_nuevos_futuros = pd.concat(X_nuevos_futuros, ignore_index=True)  # Concatenar las filas generadas
            y_pred_nuevos_120 = best_rf.predict(X_nuevos_futuros)

            # Evaluar el modelo con los últimos 120 valores conocidos
            mse_rf = mean_squared_error(y_test_last_120, y_pred_last_120)
            r2_rf = r2_score(y_test_last_120, y_pred_last_120)
            st.write(f'Random Forest con validación cruzada - MSE: {mse_rf}, R2: {r2_rf}')

            # Gráfico de comparación de valores reales y predicciones (últimos 120 valores conocidos)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_test_last_120['date'], y=y_test_last_120, name='Valores Reales'))
            fig.add_trace(go.Scatter(x=df_test_last_120['date'], y=y_pred_last_120, mode='lines', name='Predicciones', line=dict(color='red')))

            # Agregar las predicciones de los nuevos 120 datos (5 días futuros)
            fig.add_trace(go.Scatter(x=future_dates, y=y_pred_nuevos_120, mode='lines', name='Pronostico a 5 dias', line=dict(color='blue', dash='dash')))

            fig.update_layout(title=f"Comparación de {componente_estacion}: Últimos 120 datos y Predicciones a 5 días futuros",
                              xaxis_title='Fecha', yaxis_title='Concentración')
            st.plotly_chart(fig)

            # Tabla con las predicciones de los nuevos 120 datos (5 días futuros)
            df_resultados = pd.DataFrame({
                'date': df_test_last_120['date'],
                'Valores Reales': y_test_last_120.values,
                'Predicciones': y_pred_last_120,
                'Error': (y_test_last_120.values - y_pred_last_120)
            })
            st.write("Resultados de los ultimos dias:")
            st.dataframe(df_resultados.iloc[::-1], width=1000, height=500)

        else:
            st.error(f"La columna {componente_estacion} no se encuentra en el dataset.")
    else:
        st.write("Selecciona un componente y estación en el sidebar.")
