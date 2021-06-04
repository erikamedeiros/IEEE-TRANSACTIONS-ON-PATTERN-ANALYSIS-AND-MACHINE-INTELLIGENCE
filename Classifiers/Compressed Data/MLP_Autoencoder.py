from numpy.random import seed
seed(0)

# evaluate logistic regression on encoded input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from numpy import mean
import pandas as pd
import numpy as np
import time

#create dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]

X = dataset
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])
t = MinMaxScaler(feature_range = (0, 1))

# define oversampling strategy
oversample = SMOTE(random_state=0)

# load the model from file
encoder = load_model('encoder.h5', compile='false')
MLP_model = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0)

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = StratifiedKFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

results_mlp=list()
acc_MLP=list()
maior_acuracia_treino_MLP=0

inicio = time.process_time()
# split into train test sets
for i in range(1, 31):
    print('Interação: ', i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=i)
    
    # fit and apply the transform of oversample
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    # scale data
    X_train = t.fit_transform(X_train)
    X_test = t.transform(X_test)    
    
    # encode the train dat
    X_train_encode = encoder.predict(X_train)
    # encode the test data
    X_test_encode = encoder.predict(X_test)
        
    #----------------------------------------------------------------MLP
    #evaluate the model
    #preparar o modelo para ser validado no kfold
    scores_mlp = evaluate_model(X_train_encode, y_train, MLP_model)
    results_mlp.append(mean(scores_mlp))
    
    # fit the model on the training set
    MLP_model.fit(X_train_encode, y_train)
    # make predictions on the test set
    predict_MLP = MLP_model.predict(X_test_encode)
    # calculate classification accuracy
    acc_MLP.append(accuracy_score(y_test, predict_MLP))
    
    aux_MLP = max(float(results_mlp) for results_mlp in results_mlp)
    if maior_acuracia_treino_MLP < aux_MLP:
        maior_acuracia_treino_MLP = aux_MLP
        x_teste_MC_MLP = X_test_encode
        y_teste_MC_MLP = y_test
        best_MLP = MLP_model

fim = time.process_time()
print('Time Processing: ', fim-inicio)

#Criando a matriz de confusão de cada modelo
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(best_MLP, x_teste_MC_MLP, y_teste_MC_MLP, display_labels=["2-", "3", "2", "2+"])
plt.show()

print("--------------------------------------------------------------")
print ("Média da acurácia do MLP nos dados de treino {:.4f}%.".format(mean(results_mlp)*100))
print("Desvio padrao do MLP no dados de treino: ", np.std(results_mlp))
print ("A acurácia da predição do MLP foi de {:.4f}%.".format(mean(acc_MLP)*100))
print("Desvio padrao na predição do MLP: ", np.std(acc_MLP))


