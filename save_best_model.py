#Criando a minha primeira rede neural com keras
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

#fixando a seed aleatoria para manter a reprodutibilidade
seed = 7
numpy.random.seed(seed)

#carregando o dataset
dataset = numpy.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")

#quebrando a entrada em x e a saida em y
X = dataset[:,0:8]
Y = dataset[:,8]

#criando o modelo
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#compilando o modelo
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

#criando o callback para salvar o melhor modelo
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')
callback_list = [checkpoint]

#ajustando o modelo
model.fit(X, Y, nb_epoch=150, batch_size=10, callbacks=callback_list)

#estimando a performance do modelo
scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))