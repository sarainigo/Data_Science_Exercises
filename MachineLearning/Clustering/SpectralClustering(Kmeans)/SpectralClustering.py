#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:34:42 2020

@author: Sara
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import math
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn
from numpy import linalg 

df = pd.read_csv('FourCircle.csv')
print(df)

random_state = 60616

x = np.array(df['x'])
y = np.array(df['y'])


# a) plot y vs x

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot')
plt.show()


# b) K-mean algorithm
data = np.column_stack((x,y))
kmeans = cluster.KMeans(n_clusters=4, random_state = 0).fit(data)
print("Cluster Assignment:", kmeans.labels_)
print("Cluster Centroids:", kmeans.cluster_centers_)

plt.scatter(x, y, c=kmeans.labels_.astype(float))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot before K-mean clustering')
plt.show()


# c), d) Nearest neighbor algorithm

# n_neigh neighbors
n_neigh = 10
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = n_neigh,
   algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(data)
d3, i3 = nbrs.kneighbors(data)
print('Distance to the nearest neighbors: '+str(d3))
print('Which are the nearest neihbors: '+str(i3))

# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(data)

nObs = 1440

# Create the Adjacency matrix
Adjacency = np.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

print ('Adjacency matrix: '+str(Adjacency))

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())

# Create the Degree matrix
Degree = np.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

print ('Degree matrix: '+str(Degree))

# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency

print ('Laplacian matrix: '+str(Lmatrix))

# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix)

# Series plot of the smallest five eigenvalues to determine the number of clusters
sequence = np.arange(1,6,1) 
plt.plot(sequence, evals[0:5,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.xticks(sequence)
plt.grid("both")
plt.show()

# Series plot of the smallest twenty eigenvalues to determine the number of neighbors
sequence = np.arange(1,21,1) 
plt.plot(sequence, evals[0:20,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid("both")
plt.xticks(sequence)
plt.show()

print ('Eigenvalues: '+str(evals[0])+', '+str(evals[1])+', '+str(evals[2])+', '+str(evals[3]))


# e) K-mean algorithm on eigenvectors of “practically” zero eigenvalues.

# Inspect the values of the selected eigenvectors 
Z = evecs[:,[0,3]]
print('Mean: ', Z[[0]].mean(),'Standar deviation: ', Z[[0]].std())
print('Mean: ', Z[[1]].mean(),'Standar deviation: ', Z[[1]].std())
print('Mean: ', Z[[2]].mean(),'Standar deviation: ', Z[[2]].std())
print('Mean: ', Z[[3]].mean(),'Standar deviation: ', Z[[3]].std())


# Perform 4-cluster K-mean on the first four eigenvectors
kmeans_spectral = cluster.KMeans(n_clusters = 4, random_state = 0).fit(Z)
SpectralCluster = kmeans_spectral.labels_

plt.scatter(df['x'], df['y'], c = SpectralCluster)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
