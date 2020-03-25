#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:29:39 2020

@author: Sara
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import math
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

df = pd.read_csv('cars.csv')

list_of_tuples = list(zip(df['Type'], df['Origin'], df['DriveTrain'], df['Cylinders']))
df_2 = pd.DataFrame(list_of_tuples, columns = ['Type', 'Origin', 'DriveTrain', 'Cylinders'])  

'''del df['Make']
del df['Model']
del df['MSRP']
del df['Invoice']
del df['EngineSize']
del df['Horsepower']
del df['MPG_City']
del df['MPG_Highway']
del df['Weight']
del df['Wheelbase']
del df['Length']'''


# a) The frequencies of the categorical feature Type

values, counts = np.unique(df_2['Type'], return_counts=True)
print('Type: ')
print('Values: '+ str(values))
print('Frequency: '+ str(counts))


# b) The frequencies of the categorical feature DriveTrain

values, counts = np.unique(df_2['DriveTrain'], return_counts=True)
print('DriveTrain: ')
print('Values: '+ str(values))
print('Frequency: '+ str(counts))


# c) The distance between Origin = ‘Asia’ and Origin = ‘Europe’

values, counts = np.unique(df_2['Origin'], return_counts=True)
print('Origin: ')
print('Values: '+ str(values))
print('Frequency: '+ str(counts))

distance = 1/counts[0] + 1/counts[1]
print('Distance between Origins: Asia, Europe = '+str(distance))


# d) The distance between Cylinders = 5 and Cylinders = Missing

df_2['Cylinders'].fillna(0, inplace=True)

values, counts = np.unique(df_2['Cylinders'], return_counts=True)
print('Cylinders: ')
print('Values: '+ str(values))
print('Frequency: '+ str(counts))

distance = 1/counts[0] + 1/counts[3]
print('Distance between Cylinders: 5, Missing = '+str(distance))


# e) K-modes method

# K-modes algorithm

km = KModes(n_clusters=3, init='Huang',n_init=5, verbose=1)

clusters = km.fit_predict(df_2)

# Print the cluster centroids
print('Centroids of the 3 clusters: '+str(km.cluster_centroids_))

# Observations in each cluster
values, counts = np.unique(clusters, return_counts=True)
print('Observations in each cluster: ')
print('Clusters: '+ str(values))
print('No obs: '+ str(counts))


# f) Frequency distribution table for each cluster

df_2['Clustering'] = clusters
print(clusters)

# Cluster 0

df_cluster0 = df_2[df_2['Clustering'] == 0]
values, counts = np.unique(df_cluster0['Origin'], return_counts=True)
print('Cluster 0: ')
print('Values: '+ str(values))
print('Frequency: '+ str(counts))

# Cluster 1

df_cluster1 = df_2[df_2['Clustering'] == 1]
values, counts = np.unique(df_cluster1['Origin'], return_counts=True)
print('Cluster 1: ')
print('Values: '+ str(values))
print('Frequency: '+ str(counts))

# Cluster 2

df_cluster2 = df_2[df_2['Clustering'] == 2]
values, counts = np.unique(df_cluster2['Origin'], return_counts=True)
print('Cluster 2: ')
print('Values: '+ str(values))
print('Frequency: '+ str(counts))
