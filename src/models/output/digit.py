from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
def get_mnist_data():
    return mnist.load_data()
(x_train, y_train), (x_test, y_test) = get_mnist_data()

model = KNeighborsClassifier(3)
model.fit(x_train.reshape(60000,28*28)/255.0, y_train)
y_pred = model.predict(x_test.reshape(10000,28*28)/255.0)
accuracy = accuracy_score(y_test, y_pred)
def digito( image ):
    return model.predict(image)
    