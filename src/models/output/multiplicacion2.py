from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

# (x_train, y_train), (x_test, y_test) = get_mnist_data()

model = KNeighborsClassifier(3)
# model.fit(x_train.reshape(60000,28*28)/255.0, y_train)
# y_pred = model.predict(x_test.reshape(10000,28*28)/255.0)
# accuracy = accuracy_score(y_test, y_pred)
def entrenar_multiplicacion2():
    data=[]
    labels=[]
    path = Path('src/models/input/multiplicacion2')
    for item in path.iterdir():
        if item.is_file():
            data.append(np.load(item))
            if(len(item.name.split('_'))>2):
                labels.append(0)
            else:
                labels.append(1)
    path = Path('src/models/input/suma')
    for item in path.iterdir():
        if item.is_file():
            if(len(item.name.split('_'))==2):
                data.append(np.load(item))
                labels.append(0)
    path = Path('src/models/input/multiplicacion1')
    for item in path.iterdir():
        if item.is_file():
            if(len(item.name.split('_'))==2):
                data.append(np.load(item))
                labels.append(0)
    path = Path('src/models/input/resta')
    for item in path.iterdir():
        if item.is_file():
            if(len(item.name.split('_'))==2):
                data.append(np.load(item))
                labels.append(0)
    path = Path('src/models/input/division1')
    for item in path.iterdir():
        if item.is_file():
            if(len(item.name.split('_'))==2):
                data.append(np.load(item))
                labels.append(0)
    path = Path('src/models/input/division2')
    for item in path.iterdir():
        if item.is_file():
            if(len(item.name.split('_'))==2):
                data.append(np.load(item))
                labels.append(0)
    datos=np.array(data)
    resultados=np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(datos, resultados, test_size=0.1)
    model.fit(x_train.reshape(-1,28*28)/255.0, y_train)
    y_pred = model.predict(x_test.reshape(-1,28*28)/255.0)
    accuracy = accuracy_score(y_test, y_pred)
    return data, labels , accuracy
def reconocer_multiplicacion2( image ):
    return model.predict(image)
    