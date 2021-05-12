#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#create dataset
dataset = pd.read_csv('datasetv3.csv', sep=';')  
del dataset["imgName"]
#del dataset['Y']
#del dataset["medianG"]
#del dataset["medianB"]
#del dataset["valueHsv"]
#del dataset["intensityHsi"]

#rotulos do v2
X = dataset
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])

encoder = load_model('encoder.h5', compile='false')

#tunign LR
def tuning_LR(x,y):
    #Criacao do grid para o LR
    valores_solver = np.array(['newton-cg', 'lbfgs', 'liblinear'])
    valores_penalty = np.array(['none', 'l1', 'l2', 'elasticnet'])
    valores_C = np.array([100, 10, 1.0, 0.1, 0.01])
    valores_grid = {'solver': valores_solver, 'penalty': valores_penalty, 'C': valores_C}
    
    #chamando o modelo
    modelo = LogisticRegression()
    #Criando os grids
    gridLR = GridSearchCV(estimator= modelo, param_grid = valores_grid, cv = 2, n_jobs=-1)
    gridLR.fit(x, y)
    
    return gridLR

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = KFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

results_lr=list()
results_val_lr=list()
accuracy_predict_lr=list()
maior_acuracia_treino_LR = 0

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
    LR_model = tuning_LR(x_treino, y_treino)
            
    #preparar o modelo para ser validado no kfold
    scores_lr = evaluate_model(x_treino, y_treino, LR_model)
    results_lr.append(mean(scores_lr))
    ############################################################
      
    #Acurácia nos dados de validação
    results_val_lr.append(mean(LR_model.cv_results_['mean_test_score'])) 
    #################################################################
    
    #Make predictions
    LR_model.fit(x_treino, y_treino)
    predict_LR = LR_model.predict(x_teste)
    accuracy_predict_lr.append(accuracy_score(y_teste, predict_LR))
 
   ####################################################################
    
    #Gravando a base de dados de treino da melhor predicao para a MC
    aux_LR = max(float(results_lr) for results_lr in results_lr)
    if maior_acuracia_treino_LR < aux_LR:
        maior_acuracia_treino_LR = aux_LR
        x_teste_MC_LR = x_teste
        y_teste_MC_LR = y_teste
        best_LR = LR_model

#Criando a matriz de confusão de cada modelo
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_LR, x_teste_MC_LR, y_teste_MC_LR, display_labels=["2-", "3", "2", "2+"])
plt.show()

#Dados de treino
print ("Média da acurácia do LR nos dados de treino {:.4f}%.".format(mean(results_lr)*100))
print("Desvio padrao do LR no dados de treino: ", np.std(results_lr))

#Dados de validacao 
print("--------------------------------------------------------------")
print ("A acurácia da validacao do LR foi de {:.4f}%.".format(mean(results_val_lr)*100))
print("Desvio padrao nos dados de validação do LR: ", np.std(results_val_lr))

#Dados de teste
print("--------------------------------------------------------------")
print ("A acurácia da predição do LR foi de {:.4f}%.".format(mean(accuracy_predict_lr)*100))
print("Desvio padrao na predição do LR: ", np.std(accuracy_predict_lr))