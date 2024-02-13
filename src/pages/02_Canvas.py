
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st
from keras.datasets import mnist
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import os, os.path
from  models.output.suma import entrenar_suma,reconocer_suma
from  models.output.resta import entrenar_resta,reconocer_resta
from  models.output.multiplicacion1 import entrenar_multiplicacion1,reconocer_multiplicacion1
from  models.output.multiplicacion2 import entrenar_multiplicacion2,reconocer_multiplicacion2
from  models.output.division1 import entrenar_division1,reconocer_division1
from  models.output.division2 import entrenar_division2,reconocer_division2
im = Image.open("favicon.ico")

st.set_page_config(
    "EsKape Room",
    im,
    initial_sidebar_state="expanded",
    layout="wide",
)

if "number" not in st.session_state:
    st.session_state["number"] = 0


@st.cache_data
def get_mnist_data():
    return mnist.load_data()


def transform_image_to_mnist(image):
    # Check if the image has 4 channels (RGBA)
    # st.write("Transform > Dimensiones de imagen de entrada")
    # st.write(image.shape)
    if image.shape[2] == 4:
        # Remover el canal alpha
        # st.write("Transform > Remover canal alpha")
        image = image[:, :, :3]
        # st.write(image.shape)

    # Convertir imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # st.write("Transform > Conversion escala de grises")
    # st.write(gray_image.shape)
    
    # Undersampling de la imagen de INPUTxINPUT a 28x28
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

    # Preprocesamiento de la imagen para incrementar contraste
    equalized_image = cv2.equalizeHist(resized_image)
    
    # Imprimir dimensiones de salida
    # st.write("Transform > Dimensiones imagen de salida")
    # st.write(resized_image.shape)

    # Retornamos la imagen transformada de INPUTxINPUT a 28x28 y la imagen con contraste
    return resized_image, equalized_image

def mnist_dataset_viewer(x_train, y_train, x_test, y_test):
    st.header("Sección de mnist")
    option = st.sidebar.selectbox(
        "De cuál dataset quieres ver la imagen?", ("train", "test")
    )

    if option == "train":
        st.session_state["number"] = st.sidebar.slider(
            "Índice de la imagen en entrenamiento", 0, x_train.shape[0], 0
        )
        st.image(x_train[st.session_state["number"]], channels="gray")
    else:
        st.session_state["number"] = st.sidebar.slider(
            "Índice de la imagen en prueba", 0, x_test.shape[0], 0
        )
        st.image(x_test[st.session_state["number"]], channels="gray")

    st.write("Shape of mnist image")
    st.write(x_train[st.session_state["number"]].shape)

def play_canvas():
    # Cómo leer los datos de Keras
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    
    # Creando variables del sidebar
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    st.header("Escriba el operador")
    with st.container():
        st.markdown("#")
        st.markdown("#")
        operator_1 = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=None,
            update_streamlit=realtime_update,
            height=100,
            width=100,
            drawing_mode="freedraw",
            point_display_radius=0,
            key="operator_1",
        )

    if operator_1.image_data is not None:
        

        # st.write("Matriz asociada al operador")
        # st.write(operator_1.image_data)

        # st.write("Dimensiones del operador")
        # st.write(operator_1.image_data.shape)

        # st.write("Transformando operador")

        image_mnist_op, image_mnist_op_eq = transform_image_to_mnist(
            operator_1.image_data
        )

        # st.write("Exponente transformado: ")
        # # Display the image with Streamlit
        # st.image(image_mnist_op, channels="gray", caption="Grayscale Image")

        #st.write("Exponente transformado: ")
        # # Display the image with Streamlit
        #st.image(image_mnist_op_eq, channels="gray", caption="Grayscale Image")

        #st.write("Matriz asociada al exponente transformado")
        #st.write(image_mnist_op_eq)
        
        
        matrix=[]
        index=[]
        st.write("operador optimizado y ecualizado: ")
        st.image(image_mnist_op_eq)
        [add,subb,plus,div]=st.tabs(["suma", "resta", "multiplicacion", "division"])
        
        with add:
            col1,col2=st.columns(2)
            matrix.clear()
            index.clear()
            with col1:
                
                st.subheader("Sección de suma")
                if st.button("Entrenar suma"):
                    data,lavel,acuaracy=entrenar_suma()
                    data=np.array(data)
                    lavel=np.array(lavel)
                    st.write(data.shape)
                    st.write(lavel.shape)
                    st.write(acuaracy)
                if st.button('Save + SI'):
                    for _ in range(4):
                        path = Path('src/models/input/suma')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/suma') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/suma/train_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/suma/train_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button('Save + NO'):
                    for _ in range(4):
                        path = Path('src/models/input/suma')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/suma') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/suma/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/suma/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button("Reconocer suma"):
                    res=reconocer_suma(image_mnist_op_eq.reshape(-1,28*28)/255.0)
                    st.write(res)
            with col2:
                for i in range(len(matrix)):
                    st.write(index[i])
                    st.write(matrix[i])
        with plus:
            matrix.clear()
            index.clear()
            col1,col2=st.columns(2)
            with col1:
                st.subheader("Sección de multiplicacion")
                if st.button("Entrenar multiplicacion *"):
                    data,lavel,acuaracy=entrenar_multiplicacion1()
                    data=np.array(data)
                    lavel=np.array(lavel)
                    st.write(data.shape)
                    st.write(lavel.shape)
                    st.write(acuaracy)
                if st.button('Save * SI'):
                    for _ in range(4):
                        path = Path('src/models/input/multiplicacion1')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/multiplicacion') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/multiplicacion1/train_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/multiplicacion1/train_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button('Save * NO'):
                    for _ in range(4):
                        path = Path('src/models/input/multiplicacion1')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/multiplicacion') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/multiplicacion1/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/multiplicacion1/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button("Reconocer multiplicacion *"):
                    res=reconocer_multiplicacion1(image_mnist_op_eq.reshape(-1,28*28)/255.0)
                    st.write(res)
                if st.button("Entrenar multiplicacion x"):
                        data,lavel,acuaracy=entrenar_multiplicacion2()
                        data=np.array(data)
                        lavel=np.array(lavel)
                        st.write(data.shape)
                        st.write(lavel.shape)
                        st.write(acuaracy)
                if st.button('Save x SI'):
                    
                    for _ in range(4):
                        path = Path('src/models/input/multiplicacion2')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/multiplicacion') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/multiplicacion2/train_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/multiplicacion2/train_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button('Save x NO'):
                    for _ in range(4):
                        path = Path('src/models/input/multiplicacion2')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/multiplicacion') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/multiplicacion2/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/multiplicacion2/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button("Reconocer multiplicacion x"):
                    res=reconocer_multiplicacion2(image_mnist_op_eq.reshape(-1,28*28)/255.0)
                    st.write(res)
            with col2:
                for i in range(len(matrix)):
                    st.write(index[i])
                    st.write(matrix[i])
        with subb:
            col1,col2=st.columns(2)
            matrix.clear()
            index.clear()
            with col1:
                st.subheader("Sección de resta")
                if st.button("Entrenar resta"):
                        data,lavel,acuaracy=entrenar_resta()
                        data=np.array(data)
                        lavel=np.array(lavel)
                        st.write(data.shape)
                        st.write(lavel.shape)
                        st.write(acuaracy)
                if st.button('Save - SI'):
                    
                    for _ in range(2):
                        path = Path('src/models/input/resta')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/resta') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/resta/train_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/resta/train_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))     
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_180)
                    image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                    for _ in range(2):
                        path = Path('src/models/input/resta')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/resta') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/resta/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/resta/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))     
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_180)
                if st.button('Save - NO'):
                    for _ in range(2):
                        path = Path('src/models/input/resta')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/resta') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/resta/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/resta/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))     
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_180)
                if st.button("Reconocer resta"):
                    res=reconocer_resta(image_mnist_op_eq.reshape(-1,28*28)/255.0)
                    st.write(res)
            with col2:
                for i in range(len(matrix)):
                    st.write(index[i])
                    st.write(matrix[i])
        with div:
            col1,col2=st.columns(2)
            matrix.clear()
            index.clear()
            with col1:
                st.subheader("Sección de division")
                if st.button("Entrenar division /"):
                        data,lavel,acuaracy=entrenar_division1()
                        data=np.array(data)
                        lavel=np.array(lavel)
                        st.write(data.shape)
                        st.write(lavel.shape)
                        st.write(acuaracy)
                if st.button('Save / SI'):
                    for _ in range(4):
                        path = Path('src/models/input/division1')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/division') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/division1/train_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/division1/train_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button('Save / NO'):
                    for _ in range(4):
                        path = Path('src/models/input/division1')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/division') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/division1/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/division1/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                if st.button("Reconocer division /"):
                    res=reconocer_division1(image_mnist_op_eq.reshape(-1,28*28)/255.0)
                    st.write(res)
                if st.button("Entrenar division %"):
                        data,lavel,acuaracy=entrenar_division2()
                        data=np.array(data)
                        lavel=np.array(lavel)
                        st.write(data.shape)
                        st.write(lavel.shape)
                        st.write(acuaracy)
                if st.button('Save % SI'):
                    for _ in range(2):
                        path = Path('src/models/input/division2')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/division') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/division2/train_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/division2/train_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_180)
                    image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_90_CLOCKWISE)
                    for _ in range(2):
                        path = Path('src/models/input/division2')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/division') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/division2/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/division2/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_180)
                if st.button('Save % NO'):
                    for _ in range(2):
                        path = Path('src/models/input/division2')
                        n=sum(1 for item in path.iterdir() if item.is_file())
                        #n=len([name for name in os.listdir('src/models/input/division') if os.path.isfile(name)])
                        index.append(n)
                        with open("src/models/input/division2/train_0_"+str(n)+".npy",'wb') as f:
                            np.save(f,image_mnist_op_eq)
                        with open("src/models/input/division2/train_0_"+str(n)+".npy",'rb') as f:
                            matrix.append(np.load(f))   
                        image_mnist_op_eq=cv2.rotate(image_mnist_op_eq,cv2.ROTATE_180)
                if st.button("Reconocer division %"):
                    res=reconocer_division2(image_mnist_op_eq.reshape(-1,28*28)/255.0)
                    st.write(res)
            with col2:
                for i in range(len(matrix)):
                    st.write(index[i])
                    st.write(matrix[i])
        

def main():
    
    play_canvas()

if __name__ == "__main__":
    main()
