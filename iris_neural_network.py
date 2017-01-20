import numpy
import pandas as pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# inicializando o gerador de numeros aleatorios
seed = 7
numpy.random.seed(seed)

#carregando os dados do dataset
dataframe = pandas.read_csv("data/iris.data", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#hot encoding
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)
#Convertendo nuemros inteiros em variaveis de classes
dummy_y = np_utils.to_categorical(encoded_y)

#definindo o modelo da rede neural
def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    #compilando
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)

#avaliando resultados com o k-fold

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))