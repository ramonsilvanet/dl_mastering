from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D
#import matplotlib.pyplot as plt

#Baixando o dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#plotando as imgens 
#plt.subplot(221)
#plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
#plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
#plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
#plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

#plt.show()

#remodelando o dataset para 2 dimensoes
WIDTH = 28
HEIGHT = 28
CHANNELS = 1

X_train = X_train.reshape(X_train.shape[0], CHANNELS, WIDTH, HEIGHT)
X_test = X_test.reshape(X_test.shape[0], CHANNELS, WIDTH, HEIGHT);

#codificando as saidas
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

#criando um modelo com ponto de partida
model = Sequential()
model.add(Convulational2D(32, 3, 3, border='valid', input_shape=(CHANNELS, WIDTH, HEIGHT), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))