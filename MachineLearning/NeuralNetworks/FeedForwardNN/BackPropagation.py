# -*- coding: utf-8 -*-
"""Assignment1_Sara_Inigo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jrWS71xaMy4WJpbwgY1XtkdL7TnGMcgL

A) Stochastic gradient descent
"""

import numpy as np
import scipy as sp
import pandas as pd
from numpy.linalg import norm
import copy
import os

def relu(x):
    return x*(np.sign(x)+1.)/2.

def sigmoid(x):
    return 1./(1.+np.exp(-x))
  
def softmax(x):
    return np.exp(x)/sum(np.exp(x))
  
def mynorm(Z):
    return np.sqrt(np.mean(Z**2))
  
def heav(x):
    return (np.sign(x)+1.)/2.
  
def myANN(Y,Xtrain,Xpred,W01,W02,W03,b01,b02,b03):
    # Initialization of Weights and Biases
    W1 = copy.copy(W01)
    W2 = copy.copy(W02)
    W3 = copy.copy(W03)
    b1 = copy.copy(b01)
    b2 = copy.copy(b02)
    b3 = copy.copy(b03)
    # Initialize adhoc variables
    k = 1
    change = 999
    # Begin Feedforward (assume learning rate is one)
    while (change > 0.001 and k<201):
        print("Iteration", k)
        # Hidden Layer 1
        Z1 = relu(W1@Xtrain + b1)
        # Hidden Layer 2
        Z2 = sigmoid(W2@Z1 + b2)
        # Output Layer
        Yhat = softmax(W3@Z2 + b3)
        # Find cross-entropy loss
        loss = -Y@np.log(Yhat)
        print("Current Loss:",loss)
        print("Z1: ",Z1)
        print("Z2: ",Z2)
        print("Yhat: ",Yhat)
        
        # Find gradient of loss with respect to the weights
        # Output Later
        dLdb3 = Yhat - Y #
        dLdW3 = np.outer(dLdb3,Z2)
        # Hidden Layer 2
        dLdb2 = (W3.T@(dLdb3))*Z2*(1-Z2)
        dLdW2 = np.outer(dLdb2,Z1)
        # Hidden Layer 1
        dLdb1 = (W2.T@(dLdb2))*heav(W1@Xtrain + b1)
        dLdW1 = np.outer(dLdb1,Xtrain)
        
        print("dLdb3: ", dLdb3)
        print("dLdW3: ", dLdW3)
        print("dLdb2: ", dLdb2)
        print("dLdW2: ", dLdW2)
        print("dLdb1: ", dLdb1)
        print("dLdW1: ", dLdW1)
       
        
        # Update Weights by Back Propagation
        # Output Layer
        b3 -= dLdb3 #(learning rate is one)
        W3 -= dLdW3
        # Hidden Layer 2
        b2 -= dLdb2
        W2 -= dLdW2
        # Hidden Layer 1
        b1 -= dLdb1
        W1 -= dLdW1
        
        print("b3: ", b3)
        print("W3: ", W3)
        print("b2: ", b2)
        print("W2: ", W2)
        print("b1: ", b1)
        print("W1: ", W1)
        
        change = norm(dLdb1)+norm(dLdb2)+norm(dLdb3)+norm(dLdW1)+norm(dLdW2)+norm(dLdW3)
        k+= 1
        
    Z1pred = W1@Xpred + b1
    Z2pred = W2@relu(Z1pred) + b2
    Z3pred = W3@sigmoid(Z2pred) + b3
    Ypred = softmax(Z3pred)
    print("")
    print("Summary")
    print("Target Y", Y)
    print("Fitted Ytrain", Yhat)
    print("Xpred", Xpred)
    print("Fitted Ypred", Ypred)
    print("Weight Matrix 1", W1)
    print("Bias Vector 1", b1)
    print("Weight Matrix 2", W2)
    print("Bias Vector 2", b2)
    print("Weight Matrix 3", W3)
    print("Bias Vector 3", b3)

W0_1 = np.array([[0.1,0.3,0.7],[0.9,0.4,0.4]])
b_1 = np.array([1.,1.])
W0_2 = np.array([[0.4,0.3],[0.7,0.2]])
b_2 = np.array([1.,1.])
W0_3 = np.array([[0.5,0.6],[0.6,0.7],[0.3,0.2]])
b_3 = np.array([1.,1.,1.])
YY = np.array([1.,0.,0.])
X_train = np.array([0.1,0.7,0.3])
X_pred = X_train


myANN(YY,X_train,X_pred,W0_1,W0_2,W0_3,b_1,b_2,b_3)

"""B) Keras:"""

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# Create Model
model = Sequential()
model.add(Dense(2, input_dim=3, activation='relu', weights = [W0_1.T,b_1]))
model.add(Dense(2, activation='sigmoid', weights = [W0_2.T,b_2]))
model.add(Dense(3, activation='softmax', weights = [W0_3.T,b_3]))
# Compile Model
sgd = optimizers.SGD(lr=1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_crossentropy'])
model.get_weights()

# Fit the model
model.fit(X_train.reshape((1,3)), YY.reshape((1,3)), epochs=200, batch_size=1)

model.predict(X_pred.reshape((1,3)))

model.get_weights()

"""SOLUTION: Comparing the weights and biases obtained by Keras with those obtained by the procedure implemented after 200 epochs, it can be concluded that the two solutions are exactly the same."""