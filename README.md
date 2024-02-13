# Tarea 2.B: Clasificación

## Objetivo

El objetivo es de la tarea es habilitar la sección `A jugar` para que tengamos un panel como el siguiente:

![Canvas de imagen](/src/img/canvas.png)

en el cuál podamos ejecutar una operación matemática sencilla y evaluar su resultado.

### El Canvas

Tenemos entonces tres tipos de input en nuestro canvas:

![Canvas de imagen](/src/img/canvas2.png)

1. Exponentes: 3 posibles referentes a los cuadrados morados. Deben ser números del 0 al 9.
2. Operadores: 2 posibles referentes a los cuadrados azules. Explicados en la siguiente sección.
3. Números: 3 posibles referentes a los cuadrados rojos. Deben ser números del 0 al 9.

## Modelo de operadores

### Datos de entrada

El dataset **debe** ser creado por ustedes! La cantidad de imágenes, estilo, resolución y cualquier elemento que consideren relevante debe ser decidido por ustedes: sean creativos. En el notebook referente al modelo de operadores, es necesaria una sección que explique toda la toma decisiones.

### Clases del modelo

Solo vamos a usar las 4 operaciones fundamentales: suma, resta, multiplicación y división. 

En el caso de suma y resta las únicas opciones posibles son: + (ASCII Code 43) y - (ASCII Code 45), respectivamente.

En el caso de multiplicación y división tendremos 2 opciones como sigue:

### Multiplicación

Una × (ASCII Code 215) o un asterísco * (ASCII Code 42)

![Multiplicación 1](/src/img/mult1.png) ![Multiplicación 2](/src/img/mult2.png)


### División

Un slash / (ASCII code 47) o el operando convencional ÷ (ASCII code 247)

![Multiplicación 1](/src/img/div2.png) ![Multiplicación 2](/src/img/div1.png)

## Modelo de clasificación de números

### Datos de entrada

Ya disponible en el código

```python
from keras.datasets import mnist

@st.cache_data
def get_mnist_data():
    return mnist.load_data()
```
Usaremos el dataset [mnist](http://www.pymvpa.org/datadb/mnist.html) con la interfaz de [Keras](https://keras.io/getting_started/) para facilitar consistencia en todos los proyectos.

### Sobre la evaluación del pipeline

1. En la sección `notebooks` deben crear uno o dos archivos `.ipynb` donde expliquen claramente todo el pipeline que usaron para crear los modelos, explicar el proceso de preprocesamiento, creación de datos (de ser necesario) y cualquier elemento que consideren relevante.
2. Los modelos asociados a la solución principal deben haber sido vistos en clase.
3. Los modelos **extras** que quisieran agregar para comparar rendimiento pueden no haber sido vistos en clases (redes neuronales convolucionales usando PyTorch por ejemplo)


## Comentarios

### Sobre el framework de la aplicación

El código base está hecho en [Streamlit](https://streamlit.io/) un framework para desarrollar aplicaciones en Python enfocadas en datos.

La [Documentación](https://docs.streamlit.io/) de streamlit es bastante sencilla de usar y la mayoría de funcionalidades necesarias ya están implementadas.

### Cómo ejecutar la aplicación

Necesitamos solamente los siguientes comandos de `Docker compose`

1. `docker compose build` crea el contenedor
2. `docker compose up` lo ejecuta en modo desarrollador
3. `docker compose up -d` lo ejecuta modo daemon

### Sobre las operaciones

1. Asumimos que la aplicación siempre será usada por un agente honesto. No se debe validar para datos que no sean los referentes al modelo (aunque es un problema interesante de resolver)
2. Somos consistentes en la entrada de cada canvas así como en el orden de las operaciones: de izquierda a derecha y con prioridad de operadores: ^, ( *, /), (+, -).

### Sobre la parte visual

Escoger las secciones útiles de `02_Canvas.py` y crear la vista referente a cada uno de los elementos de entrada:

1. 3 Coeficientes
2. 3 exponentes
3. 2 operadores

Para luego llamar a los modelos y evaluar la función.

