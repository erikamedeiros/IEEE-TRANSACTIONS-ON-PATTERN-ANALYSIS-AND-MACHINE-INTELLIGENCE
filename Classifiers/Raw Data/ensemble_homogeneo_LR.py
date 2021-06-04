from numpy.random import seed
seed(0)

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

#create dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]

X = dataset
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])

# define oversampling strategy
oversample = SMOTE(random_state=0)

#Normalizador
normalizador = MinMaxScaler(feature_range = (0, 1))

#Criando um ENSEMBLE com os modelos variados de RF
LR_model1 = LogisticRegression(solver='newton-cg', random_state=0)
LR_model2 = LogisticRegression(solver='lbfgs', random_state=0)
LR_model3 = LogisticRegression(solver='liblinear', random_state=0)
LR_model4 = LogisticRegression(solver='sag', random_state=0)
LR_model5 = LogisticRegression(solver='saga', random_state=0)

estimators = []

estimators.append(('LR1', LR_model1))
estimators.append(('LR2', LR_model2))
estimators.append(('LR3', LR_model3))
estimators.append(('LR4', LR_model4))
estimators.append(('LR5', LR_model5))

ensemble_model = VotingClassifier(estimators)

#Listas para armazenamento
#Dados de treino
results_ensemble=list()

#Dados de teste
accuracy_predict_ensemble=list()

#Dados para a matriz de confusao
maior_acuracia_treino_ensemble = 0

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = StratifiedKFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

inicio = time.process_time()
for i in range(1, 31):
    print('Interação: ', i)
    #Divisão da base em treino (80%) e teste (20%)
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=True, stratify = y, random_state=i)
   
    # fit and apply the transform of oversample
    x_treino, y_treino = oversample.fit_resample(x_treino, y_treino)
    
    #normalização do x_treino e x_teste
    x_treino = normalizador.fit_transform(x_treino)
    x_teste = normalizador.transform(x_teste)
    
    #preparar o modelo para ser validado no kfold
    scores_ensemble =  evaluate_model(x_treino, y_treino, ensemble_model)
    results_ensemble.append(mean(scores_ensemble))
    ############################################################
      
    #Make predictions  
    ensemble_model.fit(x_treino, y_treino)
    predict_ensemble = ensemble_model.predict(x_teste)
    accuracy_predict_ensemble.append(accuracy_score(y_teste, predict_ensemble))
    ####################################################################
    
    aux_ensemble = max(float(results_ensemble) for results_ensemble in results_ensemble)
    if maior_acuracia_treino_ensemble < aux_ensemble:
        maior_acuracia_treino_ensemble = aux_ensemble
        y_teste_MC_ensemble = y_teste
        best_ensemble_homogeneo = ensemble_model
        x_teste_MC_ensemble = x_teste

fim = time.process_time()
print('Time of processing: ', fim-inicio)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_ensemble_homogeneo, x_teste_MC_ensemble, y_teste_MC_ensemble, display_labels=["2-", "3", "2", "2+"])
plt.show()

#Dados de treino
print ("Média da acurácia do ENSEMBLE HOMOGÊNEO nos dados de treino {:.4f}%.".format(mean(results_ensemble)*100))
print("Desvio padrao do ENSAMBLE HOMOGÊNEO nos dados de treino: ", np.std(results_ensemble))

#Dados de teste
print ("A acurácia da predição do ENSEMBLE HOMOGÊNEO foi de {:.4f}%.".format(mean(accuracy_predict_ensemble)*100))
print("Desvio padrao na predição do ENSEMBLE HOMOGÊNEO: ", np.std(accuracy_predict_ensemble))

#Criando um boxplot para as acuracia de treino
fig = plt.figure(figsize =(10, 7))
data = [results_ensemble]
plt.boxplot(data, showmeans=True)
plt.show()

