#Testes com RF

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#create dataset
#create dataset
dataset = pd.read_csv('datasetv3.csv', sep=';')  
del dataset["imgName"]

#del dataset['Y']
#del dataset["median"]
#del dataset["valueHsv"]
#del dataset["intensityHsi"]

#rotulos do v2
X = dataset
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])

#tunign RF
def tuning_RF(x,y):
    #Criacao do grid para o RF
    valores_n_estimators = np.array([10, 50, 100, 200])
    valores_max_features = np.array(['auto', 'sqrt', 'log2'])
    valores_max_depth = np.array([1, 3, 5, 10, 20])
    valores_min_samples_split = np.array([3, 5])
    valores_min_samples_leaf = np.array([1,2])
    valores_grid = {'n_estimators': valores_n_estimators,
                    #'max_features': valores_max_features, 
                    'max_depth': valores_max_depth,
                    #'min_samples_split': valores_min_samples_split,
                    'min_samples_leaf': valores_min_samples_leaf}
    
    #chamando o modelo
    modelo = RandomForestClassifier(random_state=0)
    #Criando os grids
    gridRF = GridSearchCV(estimator= modelo, param_grid = valores_grid, cv = 2, n_jobs=-1)
    gridRF.fit(x, y)
    
    return gridRF

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = KFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

#Plot a matrix de confusao
  
#Listas para armazenamento
#Dados de treino
results_rf=list()

#Dados de validação
results_val_RF=list()

#Dados de teste
accuracy_predict_RF=list()

#Dados para a matriz de confusao
maior_acuracia_treino_RF = 0

for seed in range(1, 31):
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('Iteração: ', seed)
    #Divisão da base em treino (80%) e teste (20%)
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=True, stratify = y, random_state=seed)
    
    #normalização do x_treino e x_teste
    normalizador = MinMaxScaler(feature_range = (0, 1))
    x_treino = normalizador.fit_transform(x_treino)
    x_teste = normalizador.fit_transform(x_teste)
   
    #chamando o tuning dos classificadores
    RF_model = tuning_RF(x_treino, y_treino)   
        
    #preparar o modelo para ser validado no kfold
    scores_rf = evaluate_model(x_treino, y_treino, RF_model)       
    results_rf.append(mean(scores_rf))
    #Para usar o mean_train_score use return_train_score=true dentro do grid
    #results_rf.append(mean(RF_model.cv_results_['mean_train_score']))
    ############################################################
      
    #Acurácia nos dados de validação
    results_val_RF.append(mean(RF_model.cv_results_['mean_test_score'])) 
    #################################################################
    
    #Make predictions
    RF_model.fit(x_treino, y_treino)
    predict_RF = RF_model.predict(x_teste)
    accuracy_predict_RF.append(accuracy_score(y_teste, predict_RF))
    
   ####################################################################
    
    #Gravando a base de dados de treino da melhor predicao para a MC
    aux_RF = max(float(results_rf) for results_rf in results_rf)
    if maior_acuracia_treino_RF < aux_RF:
        maior_acuracia_treino_RF = aux_RF
        x_teste_MC_RF = x_teste
        y_teste_MC_RF = y_teste
        best_RF = RF_model

#Criando a matriz de confusão de cada modelo
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_RF, x_teste_MC_RF, y_teste_MC_RF)
plt.show()
     
#Dados de treino
print ("Média da acurácia do RF nos dados de treino {:.4f}%.".format(mean(results_rf)*100))
print("Desvio padrao do RF no dados de treino: ", np.std(results_rf))

#Dados de validacao 
print("--------------------------------------------------------------")
print ("A acurácia da validacao do RF foi de {:.4f}%.".format(mean(results_val_RF)*100))
print("Desvio padrao nos dados de validação do RF: ", np.std(results_val_RF))

#Dados de teste
print("--------------------------------------------------------------")
print ("A acurácia da predição do RF foi de {:.4f}%.".format(mean(accuracy_predict_RF)*100))
print("Desvio padrao na predição do RF: ", np.std(accuracy_predict_RF))

'''
#Criando um boxplot para as acuracia de treino
fig = plt.figure(figsize =(10, 7))
data = [results_knn, results_svc, results_mlp]
plt.boxplot(data, showmeans=True)
plt.show()


#Criando um boxplot para as acuracia de vakudacao
fig = plt.figure(figsize =(10, 7))
data = [results_val_KNN, results_val_SVC, results_val_MLP]
plt.boxplot(data, showmeans=True)
plt.show()


#Criando um boxplot para as acuracia de teste
fig = plt.figure(figsize =(10, 7))
data = [accuracy_predict_KNN, accuracy_predict_SVC, accuracy_predict_MLP]
plt.boxplot(data, showmeans=True)
plt.show()
'''
