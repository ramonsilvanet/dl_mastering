#Criando a minha primeira rede neural com keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

#fixando a seed aleatoria para manter a reprodutibilidade
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score

seed = 7
numpy.random.seed(seed)

#carregando o dataset
dataset = numpy.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")

#quebrando a entrada em x e a saida em y
X = dataset[:,0:8]
Y = dataset[:,8]


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return  model


#criando o modelo
# create classifier for use in scikit-learn
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10)
# evaluate model using 10-fold cross validation in scikit-learn
kfold = StratifiedKFold(n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)