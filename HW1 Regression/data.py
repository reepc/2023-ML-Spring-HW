import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest,f_regression
data=pd.read_csv('./input/covid.train.csv')

x=data[data.columns[1:94]]
y=data[data.columns[94]]

bestfeatures=SelectKBest(score_func=f_regression,k=5).fit(x,y)
dfscores=pd.DataFrame(bestfeatures.scores_)
dfcolums=pd.DataFrame(x.columns)
features=pd.concat([dfcolums,dfscores],axis=1)
features.columns=['a','b']
top_best_features=features.nlargest(20,'b')
data_index=top_best_features.index
print(list(data_index))