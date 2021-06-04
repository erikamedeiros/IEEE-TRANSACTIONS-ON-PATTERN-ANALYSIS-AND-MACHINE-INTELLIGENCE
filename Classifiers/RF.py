# evaluate logistic regression on encoded input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Ensamble VotingClassifier
from numpy import mean
import pandas as pd
import numpy as np
import lightgbm as lgb
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

# load the model from file
encoder = load_model('encoder.h5', compile='false')

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = StratifiedKFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

results_rf=list()
acc_RF=list()
maior_acuracia_treino_RF=0

inicio = time.process_time()

for seed in range(1, 31):
    print('Interação: ', seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # fit and apply the transform of oversample
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    #create models
    RF_model =  RandomForestClassifier(random_state=0) 
    
    # scale data
    X_train = t.fit_transform(X_train)
    X_test = t.transform(X_test)    
    
    # encode the train dat
    X_train_encode = encoder.predict(X_train)
    # encode the test data
    X_test_encode = encoder.predict(X_test)
    
    #----------------------------------------------------------------RF
    #evaluate the model
    #preparar o modelo para ser validado no kfold
    scores_rf = evaluate_model(X_train_encode, y_train, RF_model)
    results_rf.append(mean(scores_rf))
    
    # fit the model on the training set
    RF_model.fit(X_train_encode, y_train)
    # make predictions on the test set
    predict_RF = RF_model.predict(X_test_encode)
    # calculate classification accuracy
    acc_RF.append(accuracy_score(y_test, predict_RF))
    

    aux_RF = max(float(results_rf) for results_rf in results_rf)
    if maior_acuracia_treino_RF < aux_RF:
        maior_acuracia_treino_RF = aux_RF
        x_teste_MC_RF = X_test_encode
        y_teste_MC_RF = y_test
        best_RF = RF_model

fim = time.process_time()

#Criando a matriz de confusão de cada modelo
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(best_RF, x_teste_MC_RF, y_teste_MC_RF, display_labels=["2-", "3", "2", "2+"])
plt.show()

print("--------------------------------------------------------------")
print ("Média da acurácia do RF nos dados de treino {:.4f}%.".format(mean(results_rf)*100))
print("Desvio padrao do RF no dados de treino: ", np.std(results_rf))
print ("A acurácia da predição do RF foi de {:.4f}%.".format(mean(acc_RF)*100))
print("Desvio padrao na predição do RF: ", np.std(acc_RF))
print("--------------------------------------------------------------")

