# -*- coding: utf-8 -*-
"""Exercise_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p_AW3-RLDfXABe4Qyj2qb25GXrLGzJZs

In this question the function "sensitivities" has been created, and tested for the DGP of Question 2.
"""

import random
import pandas as pd
import numpy as np
import seaborn as sns
import math
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import statsmodels.api as sm
import statsmodels.stats.diagnostic as tds
from statsmodels.api import add_constant
from scipy import stats
import matplotlib.pyplot as plt

"""DGP"""

M = 5000 # Number of sample 
np.random.seed(7) # set the seed for reproducebility of results
X = np.zeros(shape=(M,2))
X[:int(M/2),0]= np.random.randn(int(M/2))
X[:int(M/2),1]= np.random.randn(int(M/2))

X[int(M/2):,0]= -X[:int(M/2),0]
X[int(M/2):,1]= -X[:int(M/2),1]

eps = np.zeros(shape=M)
eps[:int(M/2)]= np.random.randn(int(M/2))
eps[int(M/2):]= -eps[:int(M/2)]

Y= 1.0*X[:,0] + 1.0*X[:,1] + eps

"""Creation of the model. Particularization for 3 hidden layers with 10 hidden units, with tanh activation"""

# number of hidden neurons
n=10
# number of layers
L=4
# name of activation function
activation_f = "tanh"

# with non-linear activation
def linear_NN1_model_act(l1_reg=0.0):    
    model = Sequential()
    model.add(Dense(n, input_dim=2, kernel_initializer='normal', activation=activation_f))
    
    for i in range(0,L-2):
      model.add(Dense(n, kernel_initializer='normal', activation=activation_f))
    
    model.add(Dense(1, kernel_initializer='normal')) 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    return model

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

lm = KerasRegressor(build_fn=linear_NN1_model_act, epochs=100, batch_size=10, verbose=1, callbacks=[es])

lm.fit(X,Y)

"""**SENSITIVITIES** **FUNCTION** (This function receives the number of layers and the type of activation function you want to use: relu or tanh)"""

def sensitivities(lm, X, L, activation_f):
    
    M = np.shape(X)[0]
    p = np.shape(X)[1]

    # The vector beta has beta1 and beta2, we dont compute beta0
    beta=np.array([0]*M*p, dtype='float32').reshape(M,p)
    
    W_L=np.transpose(lm.model.get_weights()[L*2-2])
    b_L=lm.model.get_weights()[L*2-1]
   
    
    # we fill beta1 and beta2 for each observation in M
    
    if activation_f == "tanh":
      
      for i in range(M):

        #initialitation when l=1
        W_l = np.transpose(lm.model.get_weights()[0])
        b_l = lm.model.get_weights()[1]
        
        Z_l = np.tanh(W_l@np.transpose(X[i,]) + b_l)
        D_l = np.diag(1-Z_l**2)
        
        J = D_l@W_l

        for l in range(2,L):
          W_l=np.transpose(lm.model.get_weights()[2*l-1-1])
          b_l=lm.model.get_weights()[2*l-1]
          
          Z_l = np.tanh(W_l@Z_l + b_l)
          D_l = np.diag(1-Z_l**2)
          
          J = (D_l@W_l)@J 

        beta[i,:]=W_L@J

      return(beta)
      
    elif activation_f == "relu":
      
      for i in range(M):

        #initialitation when l=1
        W_l = np.transpose(lm.model.get_weights()[0])
        b_l = lm.model.get_weights()[1]
              
        I_l = W_l@np.transpose(X[i,]) + b_l
        Z_l = np.maximum(I_l,0)
        D_l = np.diag((np.sign(I_l)+1.)/2.)
        
        J = D_l@W_l

        for l in range(2,L):

          W_l=np.transpose(lm.model.get_weights()[2*l-1-1])
          b_l=lm.model.get_weights()[2*l-1]

          I_l = W_l@Z_l + b_l
          Z_l = np.maximum(I_l,0)
          D_l = np.diag((np.sign(I_l)+1.)/2.)
          
          J = (D_l@W_l)@J  

        beta[i,:]=W_L@J

      return(beta)

"""**Call to the function**"""

beta=sensitivities(lm, X, L, activation_f)
  
beta_mean = np.mean(beta, axis=0)
beta_std = np.std(beta, axis=0)

beta_1_mean = beta_mean[0]
beta_1_std = beta_std[0]
beta_2_mean = beta_mean[1]
beta_2_std = beta_std[1]

print(beta_1_mean)
print(beta_1_std)
print(beta_2_mean)
print(beta_2_std)

