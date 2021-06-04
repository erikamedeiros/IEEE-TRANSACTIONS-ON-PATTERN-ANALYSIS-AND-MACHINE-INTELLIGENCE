#Referencia: https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/0.
# Friedman test
from numpy.random import seed
from scipy.stats import friedmanchisquare

import pandas as pd


# seed the random number generator
seed(0)
# generate three independent samples
dataset = pd.read_csv('dados_treino_autoencoder.csv')  

data1 = dataset['RF in Compressed Data']
data2 = dataset['LR in Compressed Data']
data3 = dataset['MLP in Compressed Data']
data4 = dataset['RF_Ensemble in Compressed Data']
data5 = dataset['LR_Ensemble in Compressed Data']
data6 = dataset['MLP_Ensemble in Compressed Data']
data7 = dataset['Hybrid_Ensemble in Compressed Data']

# compare samples
stat, p = friedmanchisquare(data1, data2, data3, data4, data5, data6, data7)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')

from autorank import autorank, create_report, plot_stats
results = autorank(dataset)
create_report(results)
plot_stats(results)

