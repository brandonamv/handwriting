from pathlib import Path
import cv2
import pandas as pd
from models.output.division1 import entrenar_division1, reconocer_division1
from models.output.division2 import entrenar_division2, reconocer_division2
from models.output.multiplicacion1 import entrenar_multiplicacion1, reconocer_multiplicacion1
from models.output.multiplicacion2 import entrenar_multiplicacion2, reconocer_multiplicacion2
from models.output.resta import entrenar_resta, reconocer_resta
from models.output.suma import entrenar_suma, reconocer_suma
import streamlit as st

from streamlit_drawable_canvas import st_canvas
from PIL import Image
from  models.output.digit import digito
import tensorflow as tf

@st.cache_resource
def entrenar():
    entrenar_division1()
    entrenar_division2()
    entrenar_suma()
    entrenar_resta()
    entrenar_multiplicacion1()
    entrenar_multiplicacion2()
im = Image.open("favicon.ico")
signos=['+','-','*','x','/','÷']
st.set_page_config(
    "EsKape Room",
    im,
    initial_sidebar_state="expanded",
    layout="wide",
)

def transform_image_to_mnist(image):
    # Check if the image has 4 channels (RGBA)
    #st.write("Transform > Dimensiones de imagen de entrada")
    #st.write(image.shape)
    if image.shape[2] == 4:
        # Remover el canal alpha
        #st.write("Transform > Remover canal alpha")
        image = image[:, :, :3]
        #st.write(image.shape)

    # Convertir imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #st.write("Transform > Conversion escala de grises")
    #st.write(gray_image.shape)
    
    # Undersampling de la imagen de INPUTxINPUT a 28x28
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)

    # Preprocesamiento de la imagen para incrementar contraste
    equalized_image = cv2.equalizeHist(resized_image)

    # Imprimir dimensiones de salida
    #st.write("Transform > Dimensiones imagen de salida")
    #st.write(resized_image.shape)

    # Retornamos la imagen transformada de INPUTxINPUT a 28x28 y la imagen con contraste
    return resized_image, equalized_image
# def explanation():
#     st.header("Página principal")

#     st.subheader("Objetivo")

#     st.write("""
#              El objetivo es de la tarea es habilitar la sección `A jugar` para que tengamos un panel como el siguiente:
#     """)

#     st.image("src/img/canvas.png")

#     st.write("""en el cuál podamos ejecutar una operación matemática sencilla.

#     Tenemos entonces tres tipos de input en nuestro canvas:""")

#     st.image("src/img/canvas2.png")

#     st.write("""1. Exponentes: 3 posibles referentes a los cuadrados morados. Deben ser números del 0 al 9.
#     2. Operadores: 2 posibles referentes a los cuadrados azules. Explicados en la siguiente sección.
#     3. Números: 3 posibles referentes a los cuadrados rojos. Deben ser números del 0 al 9.""")

#     st.subheader("Operadores")

#     st.write("""Solo vamos a usar las 4 operaciones fundamentales: suma, resta, multiplicación y división.

#     En el caso de suma y resta las únicas opciones posibles son: + (ASCII Code 43) y - (ASCII Code 45), respectivamente.

#     En el caso de multiplicación y división tendremos 2 opciones como sigue:""")

#     st.subheader("Multiplicación")

#     st.write("""Una × (ASCII Code 215) o un asterísco * (ASCII Code 42)""")

#     st.image("src/img/mult2.png")


#     st.subheader("División")

#     st.write("""Un slash / (ASCII code 47) o el operando convencional ÷ (ASCII code 247)""")

#     st.image("src/img/div1.png")


#     st.subheader("Comentarios")

#     st.subheader("Sobre las operaciones")

#     st.write("""1. Asumimos que la aplicación siempre será usada por un agente honesto. No se debe validar para datos que no sean los referentes al modelo (aunque es un problema interesante de resolver)
#     2. Somos consistentes en la entrada de cada canvas así como en el orden de las operaciones: de izquierda a derecha y con prioridad de operadores: ^, ( *, /), (+, -).""")

#     st.subheader("Sobre la parte visual")

#     st.write("""Escoger las secciones útiles de 02_Canvas.py y crear la vista referente a cada uno de los elementos de entrada:

#     1. 3 Coeficientes
#     2. 3 exponentes
#     3. 2 operadores

#     Para luego llamar a los modelos y evaluar la función.""")



def user_panel():
    # Esta sección de código debería ser st.write(Path("src/md/Objetivo.md").read_text())
    # Pero streamlit no soporta imágenes dentro de markdowns
    #explanation()

    #st.write(Path("src/md/Requerimientos.md").read_text())

    #######################################################
    #               INGRESAR CÓDIGO ACÁ                   #
    #######################################################
    # Creando variables del sidebar
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    numero_1=int(-1)
    numero_2=-1
    numero_3=-1
    exp_1=int(1)
    exp_2=1
    exp_3=1
    op1=-1
    op2=-1
    with st.container():
        (
            number_one,
            _,
            operator_one,
            number_two,
            _,
            operator_two,
            number_three,
        ) = st.columns([3, 1, 2, 3, 1, 2, 3])

        with number_one:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_1",
                )

            number_1 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_1",
            )

        with operator_one:
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
        with number_two:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_2 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_2",
                )
            number_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_2",
            )

        with operator_two:
            st.markdown("#")
            st.markdown("#")
            operator_2 = st_canvas(
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
                key="operator_2",
            )

        with number_three:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_3 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_3",
                )

            number_3 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_3",
            )
    
    col1,col2,col3=st.columns(3)
    with col1:
        st.header("Números")
        # Do something interesting with the image data and paths
        if number_1.image_data is not None:
            # st.write("Image: ")
            # st.image(number_1.image_data)

            # st.write("Dimensiones de la imagen")
            # st.write(number_1.image_data.shape)

            # st.write("Matriz asociada a la imagen")
            # st.write(number_1.image_data)

            # st.write("Transforming image")

            image_mnist, image_mnist_eq = transform_image_to_mnist(number_1.image_data)

            # st.write("Image Transformed: ")
            # # Display the image with Streamlit
            # st.image(image_mnist, channels="gray", caption="Grayscale Image")

            # st.write("Image Transformed equalized: ")
            # # Display the image with Streamlit
            # st.image(image_mnist_eq, channels="gray", caption="Grayscale Image")

            # st.write("Matriz asociada a la imagen transformada")
            # st.write(image_mnist_eq)
        
            numero_1=digito(image_mnist_eq.reshape(1,28*28)/255.0)[0]
            st.write("numero 1:"+str(numero_1))

        # if number_1.json_data is not None:
        #     objects = pd.json_normalize(
        #         number_1.json_data["objects"]
        #     )  # need to convert obj to str because PyArrow
        #     for col in objects.select_dtypes(include=["object"]).columns:
        #         objects[col] = objects[col].astype("str")
        #     st.dataframe(objects)
        if number_2.image_data is not None:
            image_mnist, image_mnist_eq = transform_image_to_mnist(number_2.image_data)
            numero_2=digito(image_mnist_eq.reshape(1,28*28)/255.0)[0]
            st.write("numero 2:"+str(numero_2))
        if number_3.image_data is not None:
            image_mnist, image_mnist_eq = transform_image_to_mnist(number_3.image_data)
            numero_3=digito(image_mnist_eq.reshape(1,28*28)/255.0)[0]
            st.write("numero 3:"+str(numero_3))
    with col2:
        st.header("Exponentes")
        if exponent_1.image_data is not None:
            image_mnist, image_mnist_eq = transform_image_to_mnist(exponent_1.image_data)
            exp_1=digito(image_mnist_eq.reshape(1,28*28)/255.0)[0]
            st.write("Exponente 1:"+str(exp_1))
        if exponent_2.image_data is not None:
            image_mnist, image_mnist_eq = transform_image_to_mnist(exponent_2.image_data)
            exp_2=digito(image_mnist_eq.reshape(1,28*28)/255.0)[0]
            st.write("Exponente 2:"+str(exp_2))
        if exponent_3.image_data is not None:
            image_mnist, image_mnist_eq = transform_image_to_mnist(exponent_3.image_data)
            exp_3=digito(image_mnist_eq.reshape(1,28*28)/255.0)[0]
            st.write("Exponente 3:"+str(exp_3))
    with col3:
        st.header("Operadores")
        entrenar()
        # if st.button("entrenar_modelos"):
        #     entrenar_division1()
        #     entrenar_division2()
        #     entrenar_suma()
        #     entrenar_resta()
        #     entrenar_multiplicacion1()
        #     entrenar_multiplicacion2()
        #     modelos_entrenados=True
        if operator_1.image_data is not None:
            image_mnist, image_mnist_eq = transform_image_to_mnist(operator_1.image_data)
            
            simbolo='Ninguno'
            # if modelos_entrenados:
            if reconocer_multiplicacion1(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op1=2
                simbolo=signos[2]
            elif reconocer_multiplicacion2(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op1=2
                simbolo=signos[3]
            elif reconocer_suma(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op1=0
                simbolo=signos[0]
            elif reconocer_division1(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op1=3
                simbolo=signos[4]
            elif reconocer_division2(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op1=3
                simbolo=signos[5]
            elif reconocer_resta(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op1=1
                simbolo=signos[1]
            st.write("Operador 1: "+simbolo)
            # else:
            #     st.subheader("Debe entrenar los modelos")
        if operator_2.image_data is not None:
            image_mnist, image_mnist_eq = transform_image_to_mnist(operator_2.image_data)
            # if modelos_entrenados:
            simbolo='Ninguno'
            # if modelos_entrenados:
            if reconocer_multiplicacion1(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op2=2
                simbolo=signos[2]
            elif reconocer_multiplicacion2(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op2=2
                simbolo=signos[3]
            elif reconocer_suma(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op2=0
                simbolo=signos[0]
            elif reconocer_division1(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op2=3
                simbolo=signos[4]
            elif reconocer_division2(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op2=3
                simbolo=signos[5]
            elif reconocer_resta(image_mnist_eq.reshape(1,28*28)/255.0)==1:
                op2=1
                simbolo=signos[1]
            st.write("Operador 2: "+simbolo)
    st.header("Operacion")
    res1=None
    res2=None
    n1=pow(numero_1,exp_1)
    n2=pow(numero_2,exp_2)
    n3=pow(numero_3,exp_3)
    operacion=r''
    if op1!=-1:
        operacion+=str(numero_1)+"^"+str(exp_1)
        if op1==0:
            res1=n1+n2
            operacion+="+"
        if op1==1:
            res1=n1-n2
            operacion+="-"
        if op1==2:
            res1=n1*n2
            operacion+="*"
        if op1==3:
            res1=n1/n2
            operacion+="/"
        operacion+=str(numero_2)+"^"+str(exp_2)

    if op2!=-1:
        if op2==0:
            res2=res1+n3
            operacion+="+"
        if op2==1:
            res2=res1-n3
            operacion+="-"
        if op2==2:
            res2=res1*n3
            operacion+="*"
        if op2==3:
            res2=res1/n3
            operacion+="/"
        operacion+=str(numero_3)+"^"+str(exp_3)+"="+str(res2)
    else:
        if op1!=-1:
            operacion+="="+str(res1)
    st.latex(operacion)
    

        
            
def main():
    user_panel()

if __name__ == "__main__":
    main()