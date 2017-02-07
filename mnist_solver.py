import numpy
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense


def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fixando o seed para reproductibilidade
seed = 7
numpy.random.seed(seed)

#Baixando o dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#normalizando a entrada de 0-255 para 0-1
X_train = X_train / 255
X_test = X_test / 255


#codificando as saidas
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=10, batch_size=200, verbose=2)

#Avaiacao final do modelo
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))