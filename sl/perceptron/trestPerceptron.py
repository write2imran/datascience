# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:15:31 2019

@author: IMRANAX
"""
import plotIris
import pandas as pd
import numpy as np

#Retrieve Iris data and extract only 100 rows
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# select setosa and versicolor
y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

plotIris.plotIrisData(X)

