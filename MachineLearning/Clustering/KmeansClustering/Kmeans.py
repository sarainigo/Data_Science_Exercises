#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:24:32 2020

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
import sklearn.cluster as cluster
from numpy import linalg 


df = pd.read_csv('ChicagoCompletedPotHole.csv')
N = df.shape[0]

df['LOG_N_POTHOLES_FILLED_ON_BLOCK'] = df['N_POTHOLES_FILLED_ON_BLOCK'].apply(lambda x : math.log(x))
df['LOG_N_DAYS_FOR_COMPLETION'] = df['N_DAYS_FOR_COMPLETION'].apply(lambda x : math.log(x+1) )

df_new = df[['LOG_N_POTHOLES_FILLED_ON_BLOCK', 'LOG_N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE']]

Clusters=[]
Silhouette=[]
TotalWCSS=[]
Elbow=[]
for i in range(2, 11):
    nClusters=i
    Clusters.append(nClusters)
    kmeans = cluster.KMeans(n_clusters=nClusters,random_state=20200304).fit(df_new.values)
    
    # Calculate Silhouette
    Silhouette.append(metrics.silhouette_score(df_new.values, kmeans.labels_))

    # Calculate WCSS
    WCSS = np.zeros(nClusters)
    nC = np.zeros(nClusters)
    for i in range(len(kmeans.labels_)):
        k = kmeans.labels_[i]
        nC[k] += 1
        diff = df_new.values[i] - kmeans.cluster_centers_[k]
        WCSS[k] += diff.dot(diff)

    W = 0
    wcss_aux = 0
    for k in range(nClusters):
        W += WCSS[k] / nC[k]
        wcss_aux += WCSS[k]
    Elbow.append(W)
    TotalWCSS.append(wcss_aux)

plt.plot(Clusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.show()

plt.plot(Clusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.show()

print('ElBowValue(4): {:.7f}; Silhoutte(4): {:.7f}'.format(Elbow[3],Silhouette[3]))
result = pd.DataFrame({'N_Cluster':Clusters,'Total_WCSS':TotalWCSS,'ElbowValue':Elbow,'Silhoutte':Silhouette})
result