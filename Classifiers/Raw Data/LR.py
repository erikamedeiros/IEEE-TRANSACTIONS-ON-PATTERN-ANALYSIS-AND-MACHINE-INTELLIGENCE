# evaluate logistic regression on encoded input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from imblearn.over_sampling import SMOTE
from numpy import mean
import pandas as pd
import numpy as np
import time

# define dataset
#create dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]

X = dataset
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])
t = MinMaxScaler(feature_range = (0, 1))

# define oversampling strategy
oversample = SMOTE(random_state=0)

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = StratifiedKFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

results_lr=list()
acc_LRC=list()
maior_acuracia_treino_LR=0

#create models
LRC_model = LogisticRegressionCV(random_state=0)

inicio = time.process_time()
# split into train test sets
for seed in range(1, 31):
    print('Interação: ', seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # fit and apply the transform of oversample
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    
    # scale data
    X_train = t.fit_transform(X_train)
    X_test = t.transform(X_test)    
    
    #------------------------------------------------------------LR
    #evaluate the model
    #preparar o modelo para ser validado no kfold
    scores_lr = evaluate_model(X_train, y_train, LRC_model)
    results_lr.append(mean(scores_lr))
    
    # fit the model on the training set
    LRC_model.fit(X_train, y_train)
    
    # make predictions on the test set
    predict_LRC = LRC_model.predict(X_test)
    # calculate classification accuracy
    acc_LRC.append(accuracy_score(y_test, predict_LRC))
    
    #Gravando a base de dados de treino da melhor predicao para a MC
    aux_LR = max(float(results_lr) for results_lr in results_lr)
    if maior_acuracia_treino_LR < aux_LR:
        maior_acuracia_treino_LR = aux_LR
        x_teste_MC_LR = X_test
        y_teste_MC_LR = y_test
        best_LR = LRC_model
        
fim = time.process_time()
print('Time of processing: ', fim-inicio)

#Criando a matriz de confusão de cada modelo
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(best_LR, x_teste_MC_LR, y_teste_MC_LR, display_labels=["2-", "3", "2", "2+"])
plt.show()

print("--------------------------------------------------------------")
print ("Média da acurácia do LR nos dados de treino {:.4f}%.".format(mean(results_lr)*100))
print("Desvio padrao do LR no dados de treino: ", np.std(results_lr))
print ("A acurácia da predição do LR foi de {:.4f}%.".format(mean(acc_LRC)*100))
print("Desvio padrao na predição do LR: ", np.std(acc_LRC))
print("--------------------------------------------------------------")
