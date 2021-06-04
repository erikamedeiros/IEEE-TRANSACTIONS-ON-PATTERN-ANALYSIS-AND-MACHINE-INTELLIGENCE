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
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
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

MLP_model1 = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0, hidden_layer_sizes = (4,))
MLP_model2 = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0, hidden_layer_sizes = (10,))
MLP_model3 = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0, hidden_layer_sizes = (20,))
MLP_model4 = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0, hidden_layer_sizes = (50,))
MLP_model5 = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0, hidden_layer_sizes = (80,))
MLP_model6 = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0, hidden_layer_sizes = (100,))

estimators = []

estimators.append(('MLP1', MLP_model1))
estimators.append(('MLP2', MLP_model2))
estimators.append(('MLP3', MLP_model3))
estimators.append(('MLP4', MLP_model4))
estimators.append(('MLP5', MLP_model5))
estimators.append(('MLP6', MLP_model6))

ensemble_model = VotingClassifier(estimators)

results_ensemble=list()
accuracy_predict_ensemble=list()
maior_acuracia_treino_ensemble = 0

def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = StratifiedKFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

inicio = time.process_time()
for seed in range(1, 31):
    print('Interação: ', seed)
    #Divisão da base em treino (80%) e teste (20%)
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=True, stratify = y, random_state=seed)

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
        x_teste_MC_ensemble = x_teste
        best_ensemble_homogeneo = ensemble_model
        
fim = time.process_time()
print('Time of processing: ', fim-inicio)    

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_ensemble_homogeneo, x_teste_MC_ensemble, y_teste_MC_ensemble)  
plt.show()

#Dados de treino
print ("Média da acurácia do ENSEMBLE HOMOGÊNEO MLP nos dados de treino {:.4f}%.".format(mean(results_ensemble)*100))
print("Desvio padrao do ENSAMBLE HOMOGÊNEO nos dados de treino: ", np.std(results_ensemble))
#Dados de teste
print ("A acurácia da predição do ENSEMBLE HOMOGÊNEO MLP foi de {:.4f}%.".format(mean(accuracy_predict_ensemble)*100))
print("Desvio padrao na predição do ENSEMBLE HOMOGÊNEO: ", np.std(accuracy_predict_ensemble))

