import numpy
from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils


seed = 7
numpy.random.seed(seed)

CHANNELS = 1
WIDTH = 28
HEIGHT = 28

#carregando dataset
(X_train, Y_train) , (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], CHANNELS, WIDTH, HEIGHT).astype('float32')
X_test = X_test.reshape(X_test.shape[0], CHANNELS, WIDTH, HEIGHT).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)
num_classes = y_test.shape[1]

def larger_model():
    model = Sequential()
    model.add(Convolution2D(30,5,5, border_mode='valid', input_shape=(CHANNELS, WIDTH, HEIGHT), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15,3,3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = larger_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))