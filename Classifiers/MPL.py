#codigo criado para tunar o numero de hidden layer size para ser usado no Ensemble homogeneo
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

'''
#create dataset
dataset = pd.read_csv('datasetv3.csv', sep=';')  
del dataset["imgName"]
#del dataset["valueHsv"]
#del dataset["intensityHsi"]
X = dataset.drop('Y', axis=1)
y = dataset["Y"]
'''
#create dataset
dataset = pd.read_csv('datasetv3.csv', sep=';')  
del dataset["imgName"]
#del dataset["valueHsv"]
#del dataset["intensityHsi"]

#rotulos do v2
X = dataset
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])


'''
del dataset['saturationHsv']
del dataset['hueHsv']
del dataset['valueHsv']

del dataset['saturationHsi']
del dataset['hueHsi']
del dataset['intensityHsi']

del dataset['lLab']
del dataset['aLab']
del dataset['bLab']
'''

#tunign SVM
def tuning_MLP(x,y):
    #Criacao do grid para o MLP
    valores_hidden_layer_sizes = np.array([3, 10, 20, 50, 80, 100])
    
    valores_grid = {'hidden_layer_sizes': valores_hidden_layer_sizes, 
                    }
    
    #chamando o modelo
    modelo = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0)
    #Criando os grids
    gridMLP = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=2, n_jobs=-1)
    gridMLP.fit(x,y)
    
    #Armazenando os valores do tuning para fins de documentacao
    MLP_hidden_layer_size.append(gridMLP.best_estimator_.hidden_layer_sizes)
    MLP_accuracy.append(gridMLP.best_score_)
    
    return gridMLP

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = KFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

#Listas para armazenamento
#Dados de treino
results_mlp=list()

#Dados de validação
results_val_MLP=list()
#results_val_ensemble=list()

#Dados de teste
accuracy_predict_MLP=list()

#Dados para a matriz de confusao
maior_acuracia_treino_MLP = 0

#Arrays para armazenar os valores do tuning
MLP_hidden_layer_size = []
MLP_accuracy = []

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
    MLP_model = tuning_MLP(x_treino, y_treino)
    
        
    #preparar o modelo para ser validado no kfold
    scores_mlp = evaluate_model(x_treino, y_treino, MLP_model)
    results_mlp.append(mean(scores_mlp))
    ############################################################
      
    #Acurácia nos dados de validação
    results_val_MLP.append(mean(MLP_model.cv_results_['mean_test_score']))
    #################################################################
    
    #Make predictions
    MLP_model.fit(x_treino, y_treino)
    predict_MLP = MLP_model.predict(x_teste)
    accuracy_predict_MLP.append(accuracy_score(y_teste, predict_MLP))
   ####################################################################
    
    #Gravando a base de dados de treino da melhor predicao para a MC
    aux_MLP = max(float(results_mlp) for results_mlp in results_mlp)
    if maior_acuracia_treino_MLP < aux_MLP:
        maior_acuracia_treino_MLP = aux_MLP
        x_teste_MC_MLP = x_teste
        y_teste_MC_MLP = y_teste
        best_MLP = MLP_model     
        
#Criando a matriz de confusão de cada modelo
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_MLP, x_teste_MC_MLP, y_teste_MC_MLP)
plt.show()

#Dados de treino
print ("Média da acurácia do MLP nos dados de treino {:.4f}%.".format(mean(results_mlp)*100))
print("Desvio padrao do MLP no dados de treino: ", np.std(results_mlp))

#Dados de validacao 
print ("A acurácia da validacao do MLP foi de {:.4f}%.".format(mean(results_val_MLP)*100))
print("Desvio padrao nos dados de validação do MLP: ", np.std(results_val_MLP))

#Dados de teste
print("--------------------------------------------------------------")
print ("A acurácia da predição do MLP foi de {:.4f}%.".format(mean(accuracy_predict_MLP)*100))
print("Desvio padrao na predição do MLP: ", np.std(accuracy_predict_MLP))


#Criando um boxplot para as acuracia de teste
fig = plt.figure(figsize =(10, 7))
data = [accuracy_predict_KNN, accuracy_predict_SVC, accuracy_predict_MLP]
plt.boxplot(data, showmeans=True)
plt.show()
