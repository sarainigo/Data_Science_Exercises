#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:12:09 2020

@author: Sara
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import math
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('Groceries.csv')
print(df)


# a) Histogram and percentiles

counts = df['Customer'].value_counts()
n = plt.hist(counts)
plt.ylabel('Customers')
plt.xlabel('Number of unique items')
plt.title('Histogram of number of uniqe iems')
plt.show()

Q1 = np.percentile(counts, 25)
print("25th percentile = "+str(Q1))
Q2 = np.percentile(counts, 50)
print("50th percentile = "+str(Q2))
Q3 = np.percentile(counts, 75)
print("75th percentile = "+str(Q3))


# b) How many itemsets can we find?  What is the largest k value ºamong our itemsets?

N = 9835
Nx = 75
Pr = Nx/N
print("Support = "+str(Pr))

ListItem = df.groupby(['Customer'])['Item'].apply(list).values.tolist() 

te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
trainData = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(trainData, min_support = Pr, use_colnames = True)
n_itemsets = frequent_itemsets['itemsets'].count()
print('Number of frequent itemsets: ' + str(n_itemsets))
largest_k = len(frequent_itemsets.iloc[n_itemsets-1]['itemsets'])
print('Largest k value among our itemsets: ' + str(largest_k))


# c) Association rules whose Confidence metrics are >= 1%.

assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print('Number of association rules with confidence metrics >= 1%: ' + str(assoc_rules.count()))


# d) Plot Support metrics (vertical) vs Confidence metrics(horizontal)

support_met = np.array(assoc_rules['support'])
conf_met = np.array(assoc_rules['confidence'])
lift_met = np.array(assoc_rules['lift'])

plt.scatter(conf_met,support_met, s = lift_met)
plt.xlabel('Confidence')
plt.ylabel('Support')
plt.title('Support metrics vs Confidence metrics')
plt.show()


# e) Rules whose Confidence metrics are greater than or equal to 60%

assoc_rules_2 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print(assoc_rules_2)