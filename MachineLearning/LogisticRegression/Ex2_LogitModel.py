#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:48:01 2020

@author: Sara
"""

import numpy as np
import scipy as sp
import scipy.stats 
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn import metrics
import statsmodels.tools.tools
import statsmodels.discrete.discrete_model
import statsmodels.api as stats

# QUESTION 13

x = [[0], [0], [0], [0], [1], [1], [1], [1], [2], [2], [2], [2], [3], [3], [3], [3], [4], [4], [4], [4]]
y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
x = stats.add_constant(x,prepend=True)
logit = stats.MNLogit(y, x)
fit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

x_test = [[0],[1],[2],[3],[4]]
x_test = stats.add_constant(x_test,prepend=True)
pred = logit.predict(params=fit.params,exog=x_test)

y_event = pred[:,1]
aux = 0
y_test = [0,0,1,1,1]
for i in range(len(y_event)):
    if y_test[i] == 1:
        aux += (1-y_event[i])**2
    else:    
        aux += (0-y_event[i])**2
ASE = aux/len(y_event)
RASE = np.sqrt(ASE)
print("Root Average Squared Error: ", RASE)