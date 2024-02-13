import os

import streamlit as st
from PIL import Image

im = Image.open("favicon.ico")
st.set_page_config(
    "EsKape Room",
    im,
    initial_sidebar_state="expanded",
    layout="wide",
)

# Función para guardar el archivo en una carpeta específica
def save_uploaded_file(uploaded_file, folder_name='src/models/output'):
    # Crear la carpeta si no existe
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Guardar el archivo en la carpeta
    file_path = os.path.join(folder_name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def cargar_modelos_ui():
    # Título de la aplicación
    st.title('Cargador de Modelos de Machine Learning')

    # Cargadores de archivos
    uploaded_file1 = st.file_uploader("Cargar archivo 1 (.joblib or .pickle)", type=['joblib', 'pickle'])
    uploaded_file2 = st.file_uploader("Cargar archivo 2 (.joblib or .pickle)", type=['joblib', 'pickle'])

    # Guardar los archivos cargados
    if uploaded_file1 is not None:
        file_path = save_uploaded_file(uploaded_file1)
        st.write(f"Modelo guardado en: {file_path}")

    if uploaded_file2 is not None:
        file_path = save_uploaded_file(uploaded_file2)
        st.write(f"Modelo guardado en: {file_path}")


def main():
    cargar_modelos_ui()

if __name__ == "__main__":

    main()