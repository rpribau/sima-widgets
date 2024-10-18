import streamlit as st
import app_norte  # Importa la lógica para la estación "norte"
import app_centro  # Importa la lógica para la estación "centro"
import app_noreste  # Importa la lógica para la estación "noreste"
import app_norte2  # Importa la lógica para la estación "norte 2"
import app_noroeste  # Importa la lógica para la estación "noroeste"
import app_noreste2  # Importa la lógica para la estación "noreste 2"
import app_noreste3  # Importa la lógica para la estación "noreste 3"
import app_noroeste2  # Importa la lógica para la estación "noroeste 2"


# Añadir opciones de navegación en el sidebar
st.sidebar.title("Navegación")
opcion = st.sidebar.selectbox("Selecciona una pestaña", [
    "Estación Centro",
    "Estación Noreste",
    "Estación Noreste 2",
    "Estación Noreste 3",
    "Estación Norte",
    "Estación Norte 2",
    "Estación Noroeste",
    "Estación Noroeste 2"
])
# Lógica para la pestaña seleccionada
if opcion == "Estación Centro":
    st.title("Estación Centro")
    app_centro.run()  # Ejecuta la app de la estación "centro"
elif opcion == "Estación Norte":
    st.title("Estación Norte")
    app_norte.run()  # Ejecuta la app de la estación "norte"
elif opcion == "Estación Noreste":
    st.title("Estación Noreste")
    app_noreste.run()  # Ejecuta la app de la estación "noreste"
elif opcion == "Estación Norte 2":
    st.title("Estación Norte 2")
    app_norte2.run()
elif opcion == "Estación Noroeste":
    st.title("Estación Noroeste")
    app_noroeste.run()
elif opcion == "Estación Noreste 2":
    st.title("Estación Noreste 2")
    app_noreste2.run()
elif opcion == "Estación Noreste 3":
    st.title("Estación Noreste 3")
    app_noreste3.run()
elif opcion == "Estación Noroeste 2":
    st.title("Estación Noroeste 2")
    app_noroeste2.run()
else:
    st.write("Opción no válida")



