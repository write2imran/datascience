# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:08:03 2019

@author: IMRANAX
"""

import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

def plotIrisData(X):

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')
    
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    
    # plt.savefig('images/02_06.png', dpi=300)
    plt.show()