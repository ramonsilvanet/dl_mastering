import numpy
from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils


#fixando o seed para reproductibilidade



CHANNELS = 1
WIDTH = 28
HEIGHT = 28

def baseline_model():
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(CHANNELS, WIDTH, HEIGHT), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


seed =  7
numpy.random.seed(seed)

#carregando dataset
(x_train, y_train) , (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], CHANNELS, WIDTH, HEIGHT).astype('float32')
x_test = x_test.reshape(x_test.shape[0], CHANNELS, WIDTH, HEIGHT).astype('float32')


# normalizando as entradas de 0-255 para 0-1
x_train = x_train / 255
x_test = x_test / 255

# Codificando a saida
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


model = baseline_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

#avaliacao final do modelo
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

