import streamlit as st
import app_norte  # Importa la lógica para la estación "norte"
import app_centro  # Importa la lógica para la estación "centro"

# Añadir opciones de navegación en el sidebar
st.sidebar.title("Navegación")
opcion = st.sidebar.selectbox("Selecciona una pestaña", ["Estación Centro", "Estación Norte"])

# Lógica para la pestaña seleccionada
if opcion == "Estación Centro":
    st.title("Aplicación - Estación Centro")
    app_centro.run()  # Ejecuta la app de la estación "centro"
elif opcion == "Estación Norte":
    st.title("Aplicación - Estación Norte")
    app_norte.run()  # Ejecuta la app de la estación "norte"
