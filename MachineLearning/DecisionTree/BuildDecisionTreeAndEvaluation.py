#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:50:21 2020

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

df = pd.read_csv('claim_history.csv')
# print(df)
# df.info()

# We want to predict the usage of a car with a decision tree model.

# QUESTION 1
print("QUESTION 1")

# Data partition
df = df[['CAR_TYPE', 'OCCUPATION', 'EDUCATION','CAR_USE']]
df_train , df_test = train_test_split(df, test_size=0.25, random_state=60616, stratify = df['CAR_USE'])
x_train = df_train[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
x_test = df_test[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
y_train = df_train['CAR_USE']
y_test = df_test['CAR_USE']

# a) frequency table of target variable in the Training partition

nTotal_train = len(x_train)

print('a) frequency table of Training:')
crossTable = pd.crosstab(index = y_train, columns = ["Count"], margins = True, dropna = False)
crossTable['Percent'] = 100 * (crossTable['Count'] / nTotal_train)
crossTable = crossTable.drop(columns = ['All'])
print(crossTable)

# b) frequency table of target variable in the Test partition

nTotal_test = len(x_test)

print('b) frequency table of Test:')
crossTable = pd.crosstab(index = y_test, columns = ["Count"], margins = True, dropna = False)
crossTable['Percent'] = 100 * (crossTable['Count'] / nTotal_test)
crossTable = crossTable.drop(columns = ['All'])
print(crossTable)

# c) probability that an observation is in Training partition given CAR_USE = Commercial

commercial_training = 2842/(2842+947)
print('c) Probability of Train, with CAR_USE = Commercial : '+str(commercial_training))

# d) probability that an observation is in Test partition given CAR_USE = Private

private_test = 1629/(1629+4884)
print('d) Probability of Test, with CAR_USE = Private : '+str(private_test))
print()
print('-----------------------------------------------------------------')
print()


# QUESTION 2
print("QUESTION 2")

# Decision tree

def calculate_entropy(comb, n_comb, values, predictor, df_new, total_private, total_commercial, total):
    
    vector_entropys = []
    for i in range(0, n_comb):
        df_in_comb = df_new.where(df_new[predictor].isin(comb[i]))
        df_in_comb = df_in_comb.groupby("CAR_USE").count()[predictor]
        
        # branch 1 entropy
        try:
            comb_private = df_in_comb['Private']
        except KeyError:
            comb_private = 0
        try:
            comb_commercial = df_in_comb['Commercial']
        except KeyError:
            comb_commercial = 0
        comb_total = comb_private + comb_commercial
        if comb_private>0 and comb_commercial>0:
            Entropy1 = -((comb_private/comb_total)*np.log2(comb_private/comb_total)+(comb_commercial/comb_total)*np.log2(comb_commercial/comb_total))
        else:
            Entropy1 = 0
        
        # branch 2 entropy
        comb2_private = total_private - comb_private
        comb2_commercial = total_commercial - comb_commercial
        comb2_total = comb2_private + comb2_commercial
        if comb2_private>0 and comb2_commercial>0:
            Entropy2 = -((comb2_private/comb2_total)*np.log2(comb2_private/comb2_total)+(comb2_commercial/comb2_total)*np.log2(comb2_commercial/comb2_total))
        else:
            Entropy2 = 0
            
        # split entropy
        EntropySplit = (comb_total/total)*Entropy1 + (comb2_total/total)*Entropy2
        vector_entropys.append(EntropySplit)
        
    return vector_entropys
    
    
def find_best_split_nominal(values, predictor, df_new, total_private, total_commercial, total):
    
    n_values = len(values)
    max_iterations = round(n_values/2+1)
    
    best_entropys = []
    best_splits = []
    # We are going to iterate, to find all possible combinations of splits for each group of values.
    for i in range(1, max_iterations):
        # all possible combinations when the split is with i values
        comb = list(combinations(values,i))
        n_comb = len(comb)
        # compute entropy for all this splits
        entropys = calculate_entropy(comb, n_comb, values, predictor, df_new, total_private, total_commercial, total)
        best_entropy_i = min(entropys)
        best_split_i = comb[entropys.index(min(entropys))]
        best_entropys.append(best_entropy_i)
        best_splits.append(best_split_i)
    best_entropy = min(best_entropys)
    best_split = best_splits[best_entropys.index(min(best_entropys))]
    return best_entropy, best_split


def find_best_split_ordinal(values, predictor, df_new, total_private, total_commercial, total):
    
    n_values = len(values)
       
    # We are going to create all possible combinations of values with ordinal order.
    comb = []
    split = []
    for i in range(0, n_values-1):
        split.append(values[i])
        s = split.copy()
        comb.append(s)
    n_comb = len(comb)
    
    # compute entropy for all this splits
    entropys = calculate_entropy(comb, n_comb, values, predictor, df_new, total_private, total_commercial, total)
    best_entropy = min(entropys)
    best_split = comb[entropys.index(min(entropys))]

    return best_entropy, best_split

print('----- FIRST LAYER -----')
# SPLIT OF THE FIRST LAYER
# Steps:
# 1. Find best split for each predictor   
total_private = len(df_train[df_train['CAR_USE']=='Private'])
total_commercial = len(df_train[df_train['CAR_USE']=='Commercial'])
total = total_private + total_commercial

# 1.1. Predictor CAR_TYPE. (2^(6−1)−1 = 31 possible splits)
values1 = ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van']
predictor1 = 'CAR_TYPE'

best_entropy1, best_split1 = find_best_split_nominal(values1, predictor1, df_train[[predictor1, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor1)+" is: "+str(best_entropy1))
print("The split is made with the values "+str(best_split1))
print()

# 1.2 Predictor OCCUPATION (2^(9−1)−1 = 255 possible splits)
values2 = ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown']
predictor2 = 'OCCUPATION'

best_entropy2, best_split2 = find_best_split_nominal(values2, predictor2, df_train[[predictor2, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor2)+" is: "+str(best_entropy2))
print("The split is made with the values "+str(best_split2))
print()

# 1.3 Predictor EDUCATION (5-1 = 4 possible splits)
values3 = ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors']
predictor3 = 'EDUCATION'

best_entropy3, best_split3 = find_best_split_ordinal(values3, predictor3, df_train[[predictor3, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor3)+" is: "+str(best_entropy3))
print("The split is made with the values "+str(best_split3))
print()

# 2. We will select the predictor OCCUPATION, since it's the one that reduces the most the impurity of the parent node.

# 3. Make the split, and assign observations to one of the two child nodes
best_split2_other = values2.copy()
for i in best_split2:
    best_split2_other.remove(i)
        
df_branch1 = df_train[df_train['OCCUPATION'].isin(best_split2_other)]
df_branch2 = df_train[df_train['OCCUPATION'].isin(best_split2)]

# Exercises
Entropy_RootNode = -((total_private/total)*np.log2(total_private/total) + (total_commercial/total)*np.log2(total_commercial/total))
print('a) The Entropy of the Root Node is: ', Entropy_RootNode)

print('b) The split criterion of the first layer is: ')
print('The predictor OCCUPATION')
print('The values of the first branch are: ', best_split2_other)
print('The values of the second branch are: ', best_split2)

print('c) The Entropy of the split of the first layer is: ', best_entropy2)
print()
print('----- SPLIT OF THE FIRST LAYER BY: OCCUPATION -----')
group1 = best_split2_other
group2 = best_split2
print()

print('----- SECOND LAYER, BRANCH 1 -----')
# SPLIT OF THE FIRTS BRANCH OF THE SECOND LAYER
# 1. Find best split for each predictor 
total_private = len(df_branch1[df_branch1['CAR_USE']=='Private'])
total_commercial = len(df_branch1[df_branch1['CAR_USE']=='Commercial'])
total = total_private + total_commercial

# 1.1. Predictor CAR_TYPE. (2^(6−1)−1 = 31 possible splits)
values1 = ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van']
predictor1 = 'CAR_TYPE'

best_entropy1, best_split1 = find_best_split_nominal(values1, predictor1, df_branch1[[predictor1, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor1)+" is: "+str(best_entropy1))
print("The split is made with the values "+str(best_split1))
print()

# 1.2 Predictor OCCUPATION (2^(9−1)−1 = 255 possible splits)
values2 = ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown']
predictor2 = 'OCCUPATION'

best_entropy2, best_split2 = find_best_split_nominal(values2, predictor2, df_branch1[[predictor2, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor2)+" is: "+str(best_entropy2))
print("The split is made with the values "+str(best_split2))
print()

# 1.3 Predictor EDUCATION (5-1 = 4 possible splits)
values3 = ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors']
predictor3 = 'EDUCATION'

best_entropy3, best_split3 = find_best_split_ordinal(values3, predictor3, df_branch1[[predictor3, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor3)+" is: "+str(best_entropy3))
print("The split is made with the values "+str(best_split3))
print()

# 2. We will select the predictor CAR_TYPE, since it's the one that reduces the most the impurity of the parent node.

# 3. Make the split, and assign observations to one of the two child nodes
best_split1_other = values1.copy()
for i in best_split1:
    best_split1_other.remove(i)
        
df_branch11 = df_branch1[df_branch1['CAR_TYPE'].isin(best_split1_other)]
df_branch12 = df_branch1[df_branch1['CAR_TYPE'].isin(best_split1)]

print('----- SPLIT OF THE SECOND LAYER, BRANCH 1 BY: CAR_TYPE -----')
group11 = best_split1_other
group12 = best_split1
print()

print('----- SECOND LAYER, BRANCH 2 -----')
# SPLIT OF THE SECOND BRANCH OF THE SECOND LAYER
# 1. Find best split for each predictor 
total_private = len(df_branch2[df_branch2['CAR_USE']=='Private'])
total_commercial = len(df_branch2[df_branch2['CAR_USE']=='Commercial'])
total = total_private + total_commercial

# 1.1. Predictor CAR_TYPE. (2^(6−1)−1 = 31 possible splits)
values1 = ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van']
predictor1 = 'CAR_TYPE'

best_entropy1, best_split1 = find_best_split_nominal(values1, predictor1, df_branch2[[predictor1, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor1)+" is: "+str(best_entropy1))
print("The split is made with the values "+str(best_split1))
print()

# 1.2 Predictor OCCUPATION (2^(9−1)−1 = 255 possible splits)
values2 = ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown']
predictor2 = 'OCCUPATION'

best_entropy2, best_split2 = find_best_split_nominal(values2, predictor2, df_branch2[[predictor2, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor2)+" is: "+str(best_entropy2))
print("The split is made with the values "+str(best_split2))
print()

# 1.3 Predictor EDUCATION (5-1 = 4 possible splits)
values3 = ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors']
predictor3 = 'EDUCATION'

best_entropy3, best_split3 = find_best_split_ordinal(values3, predictor3, df_branch2[[predictor3, 'CAR_USE']], total_private, total_commercial, total)
print("The smallest split entropy of "+str(predictor3)+" is: "+str(best_entropy3))
print("The split is made with the values "+str(best_split3))
print()

# 2. We will select the predictor EDUCATION, since it's the one that reduces the most the impurity of the parent node.

# 3. Make the split, and assign observations to one of the two child nodes
best_split3_other = values3.copy()
for i in best_split3:
    best_split3_other.remove(i)
        
df_branch21 = df_branch2[df_branch2['EDUCATION'].isin(best_split3_other)]
df_branch22 = df_branch2[df_branch2['EDUCATION'].isin(best_split3)]

print('----- SPLIT OF THE SECOND LAYER, BRANCH 2 BY: EDUCATION -----')
group21 = best_split3_other
group22 = best_split3
print()

print('Information of the leaves: ')
# Leaf 1
private = len(df_branch11[df_branch11['CAR_USE']=='Private'])
commercial = len(df_branch11[df_branch11['CAR_USE']=='Commercial'])
total = private + commercial
prob_commercial_leaf1 = commercial/total
if private>commercial:
    predicted_class = 'Private'
else:
    predicted_class = 'Commercial'
Entropy = -((private/total)*np.log2(private/total) + (commercial/total)*np.log2(commercial/total))
print('LEAF 1: The decision rule is: OCCUPATION isin ',group1,' and CAR_TYPE isin ',group11,'. The number of observations is: ',total,'. The number of Private cars is: ',private,'. The number of Commercial cars is: ',commercial,'. The predicted class is: ', predicted_class,'. The Entropy is: ',Entropy,'. ')    
print()

# Leaf 2
private = len(df_branch12[df_branch12['CAR_USE']=='Private'])
commercial = len(df_branch12[df_branch12['CAR_USE']=='Commercial'])
total = private + commercial
prob_commercial_leaf2 = commercial/total
if private>commercial:
    predicted_class = 'Private'
else:
    predicted_class = 'Commercial'
Entropy = -((private/total)*np.log2(private/total) + (commercial/total)*np.log2(commercial/total))
print('LEAF 2: The decision rule is: OCCUPATION isin ',group1,' and CAR_TYPE isin ',group12,'. The number of observations is: ',total,'. The number of Private cars is: ',private,'. The number of Commercial cars is: ',commercial,'. The predicted class is: ', predicted_class,'. The Entropy is: ',Entropy,'. ')    
print()

# Leaf 3
private = len(df_branch21[df_branch21['CAR_USE']=='Private'])
commercial = len(df_branch21[df_branch21['CAR_USE']=='Commercial'])
total = private + commercial
prob_commercial_leaf3 = commercial/total
if private>commercial:
    predicted_class = 'Private'
else:
    predicted_class = 'Commercial'
Entropy = -((private/total)*np.log2(private/total) + (commercial/total)*np.log2(commercial/total))
print('LEAF 3: The decision rule is: OCCUPATION isin ',group2,' and EDUCATION isin ',group21,'. The number of observations is: ',total,'. The number of Private cars is: ',private,'. The number of Commercial cars is: ',commercial,'. The predicted class is: ', predicted_class,'. The Entropy is: ',Entropy,'. ')    
print()

# Leaf 4
private = len(df_branch22[df_branch22['CAR_USE']=='Private'])
commercial = len(df_branch22[df_branch22['CAR_USE']=='Commercial'])
total = private + commercial
prob_commercial_leaf4 = commercial/total
if private>commercial:
    predicted_class = 'Private'
else:
    predicted_class = 'Commercial'
Entropy = -((private/total)*np.log2(private/total) + (commercial/total)*np.log2(commercial/total))
print('LEAF 4: The decision rule is: OCCUPATION isin ',group2,' and EDUCATION isin ',group22,'. The number of observations is: ',total,'. The number of Private cars is: ',private,'. The number of Commercial cars is: ',commercial,'. The predicted class is: ', predicted_class,'. The Entropy is: ',Entropy,'. ')    
print()
print('-----------------------------------------------------------------')
print()

# f) Kolmogorov-Smirnov statistic and the event probability cutoff value in training

# Predicted event probabilities of the training data.
df_probability_train = df_train[['OCCUPATION','CAR_TYPE','EDUCATION','CAR_USE']]
df_probability_train.loc[df_probability_train['CAR_USE'] == 'Commercial', 'CAR_USE'] = 1
df_probability_train.loc[df_probability_train['CAR_USE'] == 'Private', 'CAR_USE'] = 0
a = np.full(len(df_probability_train), 0)
df_probability_train['PROB'] = a
# Leaf1
df_probability_train.loc[df_probability_train['OCCUPATION'].isin(group1) & df_probability_train['CAR_TYPE'].isin(group11), 'PROB'] = prob_commercial_leaf1
# Leaf2
df_probability_train.loc[df_probability_train['OCCUPATION'].isin(group1) & df_probability_train['CAR_TYPE'].isin(group12), 'PROB'] = prob_commercial_leaf2
# Leaf3
df_probability_train.loc[df_probability_train['OCCUPATION'].isin(group2) & df_probability_train['EDUCATION'].isin(group21), 'PROB'] = prob_commercial_leaf3
# Leaf4
df_probability_train.loc[df_probability_train['OCCUPATION'].isin(group2) & df_probability_train['EDUCATION'].isin(group22), 'PROB'] = prob_commercial_leaf4

y_real_train = np.array(df_probability_train['CAR_USE'])
y_prob_train = np.array(df_probability_train['PROB'])
y_real_train = y_real_train.astype(int)

fpr, tpr, thresholds = metrics.roc_curve(y_real_train, y_prob_train, pos_label = 1)

# Kolmogorov Smirnov graph
print('f) Kolmogorov Smirnov')
# Draw the Kolmogorov Smirnov curve
cutoff = np.where(thresholds > 1.0, np.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o',
         label = 'True Positive',
         color = 'blue', linestyle = 'solid')
plt.plot(cutoff, fpr, marker = 'o',
         label = 'False Positive',
         color = 'red', linestyle = 'solid')
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True)
plt.show()

KS_stat = tpr[2] - fpr[2]
print('The KS statistic is:', KS_stat)

thr_train = cutoff[2]
print('The cutoff value is:', thr_train)

# QUESTION 3
print("QUESTION 3")

# TRAIN
train_private = len(df_train[df_train['CAR_USE']=='Private'])
train_commercial = len(df_train[df_train['CAR_USE']=='Commercial'])
train_total = train_private + train_commercial
train_prob_commercial = train_commercial/train_total
train_prob_private = train_private/train_total

# TEST
test_private = len(df_test[df_test['CAR_USE']=='Private'])
test_commercial = len(df_test[df_test['CAR_USE']=='Commercial'])
test_total = test_private + test_commercial
test_prob_commercial = test_commercial/test_total
test_prob_private = test_private/test_total

print('a) The threshold used is the proportion of commercial cars in the whole training data, which is:',train_prob_commercial)
print()
print('The probability of a car in the leaf 1 of being commercial is',prob_commercial_leaf1,'. Hence, the classification of this leaf is Commercial')
print()
print('The probability of a car in the leaf 2 of being commercial is',prob_commercial_leaf2,'. Hence, the classification of this leaf is Private')
print()
print('The probability of a car in the leaf 3 of being commercial is',prob_commercial_leaf3,'. Hence, the classification of this leaf is Commercial')
print()
print('The probability of a car in the leaf 4 of being commercial is',prob_commercial_leaf4,'. Hence, the classification of this leaf is Private')
print()

# Prediction of the test data with the decision tree.

df_prediction = df_test[['OCCUPATION','CAR_TYPE','EDUCATION','CAR_USE']]
df_prediction.loc[df_prediction['CAR_USE'] == 'Commercial', 'CAR_USE'] = 1
df_prediction.loc[df_prediction['CAR_USE'] == 'Private', 'CAR_USE'] = 0
a = np.full(len(df_prediction), 0)
df_prediction['PRED'] = a
df_prediction['PROB'] = a

# Leaf1
df_prediction.loc[df_prediction['OCCUPATION'].isin(group1) & df_prediction['CAR_TYPE'].isin(group11), 'PRED'] = 1
df_prediction.loc[df_prediction['OCCUPATION'].isin(group1) & df_prediction['CAR_TYPE'].isin(group11), 'PROB'] = prob_commercial_leaf1

# Leaf2
df_prediction.loc[df_prediction['OCCUPATION'].isin(group1) & df_prediction['CAR_TYPE'].isin(group12), 'PRED'] = 0
df_prediction.loc[df_prediction['OCCUPATION'].isin(group1) & df_prediction['CAR_TYPE'].isin(group12), 'PROB'] = prob_commercial_leaf2

# Leaf3
df_prediction.loc[df_prediction['OCCUPATION'].isin(group2) & df_prediction['EDUCATION'].isin(group21), 'PRED'] = 1
df_prediction.loc[df_prediction['OCCUPATION'].isin(group2) & df_prediction['EDUCATION'].isin(group21), 'PROB'] = prob_commercial_leaf3

# Leaf4
df_prediction.loc[df_prediction['OCCUPATION'].isin(group2) & df_prediction['EDUCATION'].isin(group22), 'PRED'] = 0
df_prediction.loc[df_prediction['OCCUPATION'].isin(group2) & df_prediction['EDUCATION'].isin(group22), 'PROB'] = prob_commercial_leaf4

y_real = np.array(df_prediction['CAR_USE'])
y_pred = np.array(df_prediction['PRED'])
y_prob = np.array(df_prediction['PROB'])
y_real = y_real.astype(int)
y_pred = y_pred.astype(int)


# confusion matrix
conf_mat = metrics.confusion_matrix(y_real, y_pred)
print('Confusion matrix:', conf_mat)

# a) Misclassification Rate 
accuracy = metrics.accuracy_score(y_real, y_pred)
missclass = 1 - accuracy
print('a) The misclassification Rate in the Test partition is:',missclass)
print()

# b) Misclassification Rate with KS event probability cutoff value training as threshold

# The classification in the same as in the question a)
accuracy = metrics.accuracy_score(y_real, y_pred)
missclass = 1 - accuracy
print('b) The misclassification Rate in the Test partition is:',missclass)
print() 
    
# c) RMSE
MSE = metrics.mean_squared_error(y_real,y_pred)
RMSE = math.sqrt(MSE)   
print('c) The Root Average Squared Error in the Test partition is:',RMSE)
print()
    
# d) Area Under Curve 

df_gamma = df_prediction[['CAR_USE','PROB']]

# group the predicted probabilities in 2 groups.
df_gamma = df_gamma.sort_values(by=['CAR_USE'])
df_gamma_0 = df_gamma[df_gamma['CAR_USE']==0]
df_gamma_1 = df_gamma[df_gamma['CAR_USE']==1]

# Sort the predicted probabilities in ascending order within each group 
df_gamma_0 = df_gamma_0.sort_values(by=['PROB'])
df_gamma_1 = df_gamma_1.sort_values(by=['PROB'])
df_gamma_0 = np.array(df_gamma_0['PROB'])
df_gamma_1 = np.array(df_gamma_1['PROB'])

# Table of Concordant (C), Discordant (D), and Tied (T) pairs
C = 0
D = 0
T = 0
for i in range(0,len(df_gamma_1)):
    for j in range(0,len(df_gamma_0)):
        if df_gamma_1[i] > df_gamma_0[j]: C=C+1
        if df_gamma_1[i] == df_gamma_0[j]: T=T+1
        if df_gamma_1[i] < df_gamma_0[j]: D=D+1
Pairs = C+D+T

# other way:
# AUC = metrics.roc_auc_score(y_real,y_prob)
AUC = 0.5 + 0.5 * (C - D) / Pairs

print('d) The Area Under Curve in the Test partition is:', AUC)
print()

# e) Gini Coefficient 

# other way:
# Gini = 2 * AUC - 1
Gini = (C-D)/Pairs
print('e) The Gini Coefficient in the Test partition is:', Gini)
print()

# f) Goodman-Kruskal Gamma statistic
Gamma = (C-D)/(C+D)
print('f) The Goodman-Kruskal Gamma statistic in the Test partition is:', Gamma)
print()

# g) Generate the Receiver Operating Characteristic curve

# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_real, y_prob, pos_label = 1)

# Add two dummy coordinates
OneMinusSpecificity = np.append([0], fpr)
Sensitivity = np.append([0], tpr)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])

# Draw the ROC curve

plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis("equal")
plt.title('ROC curve')
plt.show()

