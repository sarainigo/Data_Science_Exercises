#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:19:59 2020

@author: Sara
"""
import numpy 
import scipy as sp
import scipy.stats 
import pandas 
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn import metrics
import sklearn.cluster as cluster
from numpy import linalg 
import statsmodels.api as stats


df = pandas.read_csv('policy_2001.csv')

df=df[['CLAIM_FLAG','CREDIT_SCORE_BAND','BLUEBOOK_1000','CUST_LOYALTY','MVR_PTS','TIF','TRAVTIME']] 

df_train , df_test = train_test_split(df, test_size=0.25, random_state=20200304, stratify = df['CLAIM_FLAG'])

nObs = df_train.shape[0]
y = df_train['CLAIM_FLAG']
y_category = numpy.unique(y.values)


# STEP 1

# Forward Selection   
# Model 0 is CLAIM_FLAG = Intercept
X = numpy.where(y.notnull(), 1, 0)
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)

# Consider Model 1 is Origin = Intercept + CREDIT_SCORE_BAND
Credit_Score = df_train[['CREDIT_SCORE_BAND']].astype('category')
X = pandas.get_dummies(Credit_Score)
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 1 is Origin = Intercept + Bluebook_1000
X = df_train[['BLUEBOOK_1000']]
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 1 is Origin = Intercept + CUST_LOYALTY
X = df_train[['CUST_LOYALTY']]
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)



# Consider Model 1 is Origin = Intercept + MVR_PTS
X = df_train[['MVR_PTS']]
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Consider Model 1 is Origin = Intercept + TIF
X = df_train[['TIF']]
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 1 is Origin = Intercept + TRAVTIME
X = df_train[['TRAVTIME']]
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# We keep the predictor MVR_PTS, since it has the lowest p-value

# STEP 2

# Consider Model 1 is Origin = Intercept + MVR_PTS
X = df_train[['MVR_PTS']]
X = stats.add_constant(X, prepend=True)
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)

# Consider Model 2 is Origin = Intercept + MVR_PTS + CREDIT_SCORE_BAND
X = df_train[['MVR_PTS']]
Credit_Score = df_train[['CREDIT_SCORE_BAND']].astype('category')
X = X.join(pandas.get_dummies(Credit_Score))
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 2 is Origin = Intercept + MVR_PTS + BLUEBOOK_1000
X = df_train[['MVR_PTS']]
X = X.join(df_train['BLUEBOOK_1000'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 2 is Origin = Intercept + MVR_PTS + CUST_LOYALTY
X = df_train[['MVR_PTS']]
X = X.join(df_train['CUST_LOYALTY'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 2 is Origin = Intercept + MVR_PTS + TIF
X = df_train[['MVR_PTS']]
X = X.join(df_train['TIF'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 2 is Origin = Intercept + MVR_PTS + TRAVTIME
X = df_train[['MVR_PTS']]
X = X.join(df_train['TRAVTIME'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# We keep the predictor TRAVTIME, since it has the lowest p-value


# STEP 3

# Consider Model 2 is Origin = Intercept + MVR_PTS + TRAVTIME
X = df_train[['MVR_PTS']]
X = X.join(df_train['TRAVTIME'])
X = stats.add_constant(X, prepend=True)
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)

# PROBLEM, SINGULAR MATRIX
# Consider Model 3 is Origin = Intercept + MVR_PTS + TRAVTIME + CREDIT_SCORE_BAND
X = df_train[['MVR_PTS']]
X = X.join(df_train['TRAVTIME'])
Credit_Score = df_train[['CREDIT_SCORE_BAND']].astype('category')
X = X.join(pandas.get_dummies(Credit_Score))
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Consider Model 3 is Origin = Intercept + MVR_PTS + TRAVTIME + BLUEBOOK_1000
X = df_train[['MVR_PTS']]
X = X.join(df_train['TRAVTIME'])
X = X.join(df_train['BLUEBOOK_1000'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Consider Model 3 is Origin = Intercept + MVR_PTS + TRAVTIME + CUST_LOYALTY
X = df_train[['MVR_PTS']]
X = X.join(df_train['TRAVTIME'])
X = X.join(df_train['CUST_LOYALTY'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Consider Model 3 is Origin = Intercept + MVR_PTS + TRAVTIME + TIF
X = df_train[['MVR_PTS']]
X = X.join(df_train['TRAVTIME'])
X = X.join(df_train['TIF'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Missclassfication Rate

threshold = df_train[df_train['CLAIM_FLAG']==1].shape[0]/df_train.shape[0]

X_train = df_train[['MVR_PTS']]
X_train = X_train.join(df_train['TRAVTIME'])
X_train = stats.add_constant(X_train, prepend=True)
y_train = df_train[['CLAIM_FLAG']]

logit = stats.MNLogit(y_train, X_train)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

X_test = df_test[['MVR_PTS']]
X_test = X_test.join(df_test['TRAVTIME'])
X_test = stats.add_constant(X_test,prepend=True)
y_test = df_test['CLAIM_FLAG']
predProbY = thisFit.predict(X_test)

# Determine the predicted class of Y
predProbY = predProbY[1].values
predY = numpy.empty_like(y_test)
for i in range(y_test.shape[0]):
    if (predProbY[i] > threshold):
        predY[i] = 1
    else:
        predY[i] = 0
        
import sklearn.metrics as metrics
accuracy = metrics.accuracy_score(y_test, predY)
print("Misclasification Rate: {:.7f}".format(1-accuracy))
