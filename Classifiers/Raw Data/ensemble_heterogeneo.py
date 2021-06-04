from numpy.random import seed
seed(0)

# A acurácia da predição do ENSEMBLE foi de 94.9123%.
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

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import VotingClassifier

#create dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]

X = dataset
y = pd.Series([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 2, 2, 1, 3, 0, 2, 1, 3, 2, 1, 3, 0, 0, 2, 2, 1, 3, 3, 1, 3, 0, 2, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 3, 0, 3, 3, 2])

# define oversampling strategy
oversample = SMOTE(random_state=0)

KNN_model = KNeighborsClassifier()
SVC_model = SVC(random_state=0)
MLP_model = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0)
RF_model = RandomForestClassifier(random_state=0)
ETC_model = ExtraTreeClassifier(random_state=0)
NBG_model = GaussianNB()
NBB_model = BernoulliNB()
LDA_model = LinearDiscriminantAnalysis()
NC_model = NearestCentroid()
LRC_model = LogisticRegressionCV(random_state=0)
QDA_model = QuadraticDiscriminantAnalysis()
RC_model = RidgeClassifierCV()

estimators = []
#Criando um ENSEMBLE com os modelos que retornaram
estimators.append(('knn', KNN_model))
#estimators.append(('svm', SVC_model))
estimators.append(('MLP', MLP_model))
estimators.append(('RF', RF_model))
estimators.append(('ETC', ETC_model))
estimators.append(('NBG_Gaussian', NBG_model))
#estimators.append(('NBB_Bernoulli', NBB_model))
estimators.append(('LDA', LDA_model))
estimators.append(('NC', NC_model))
estimators.append(('LRC', LRC_model))
estimators.append(('QDA', QDA_model))
#estimators.append(('RC', RC_model))

ensemble_model = VotingClassifier(estimators)

results_ensemble=list()
accuracy_predict_ensemble=list()
maior_acuracia_treino_ensemble = 0


def evaluate_model(x, y, model):
	cv = StratifiedKFold(10, shuffle=True, random_state=0)
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
    normalizador = MinMaxScaler(feature_range = (0, 1))
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
        best_ensemble_heterogeneo = ensemble_model
        x_teste_MC_ensemble = x_teste
fim = time.process_time()
print('Time of processing: ', fim-inicio)  

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(best_ensemble_heterogeneo, x_teste_MC_ensemble, y_teste_MC_ensemble, display_labels=["2-", "3", "2", "2+"])
plt.show()

#Dados de treino
print ("Média da acurácia do ENSEMBLE nos dados RAW de treino {:.4f}%.".format(mean(results_ensemble)*100))
print("Desvio padrao do ENSAMBLE no dados de treino: ", np.std(results_ensemble))
#Dados de teste
print ("A acurácia da predição do ENSEMBLE foi de {:.4f}%.".format(mean(accuracy_predict_ensemble)*100))
print("Desvio padrao na predição do ENSEMBLE: ", np.std(accuracy_predict_ensemble))