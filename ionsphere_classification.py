import pandas
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler

#learning rate scheduller
def step_decay(epoch):
    initial_lrate = 0.1
    drop =0.5
    epochs_drop  = 10.0
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))


seed = 7
numpy.random.seed(seed)

#load dataset
dataframe = pandas.read_csv("data/ionosphere.data")
dataset = dataframe.values

#split dataset
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]

#encode classes values as integers
encoder = LabelEncoder();
encoder.fit(Y)

Y = encoder.transform(Y)

#create model
model = Sequential()
model.add(Dense(34, input_dim=34, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))

#compile model
epoch =50
learning_rate = 0.1
decay_rate = learning_rate / epoch
momentum = 0.9

sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#learning rate schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

#Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=epoch, batch_size=28, callbacks=callbacks_list)
