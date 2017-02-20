from keras.datasets import cifar10


#caregando os dados
from keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#normalizando os dados (entrada 0-255 para saida 0-1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)