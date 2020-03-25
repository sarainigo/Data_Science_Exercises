import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)

df = pd.read_csv('Fraud.csv')
N = df.shape[0]
print(df)

# a) Percentage of fraudulent investigations

df_fraud = df[df['FRAUD'] == 1]
df_nonfraud = df[df['FRAUD'] == 0]
n_fraud = df_fraud.shape[0]
perc_fraud = (n_fraud/N)*100
print("a) Percentage of fraudulent investigations = "+str(perc_fraud)+"%")


# b) 6 horizontal boxplots for each variable, containing 2 graphs, fraudulent and non fraudulent

total_spend = [df_fraud['TOTAL_SPEND'], df_nonfraud['TOTAL_SPEND']]
doctor_visits = [df_fraud['DOCTOR_VISITS'], df_nonfraud['DOCTOR_VISITS']]
num_claims = [df_fraud['NUM_CLAIMS'], df_nonfraud['NUM_CLAIMS']]
member_duration = [df_fraud['MEMBER_DURATION'], df_nonfraud['MEMBER_DURATION']]
optom_presc = [df_fraud['OPTOM_PRESC'], df_nonfraud['OPTOM_PRESC']]
num_members = [df_fraud['NUM_MEMBERS'], df_nonfraud['NUM_MEMBERS']]

names = ['Fraudulent', 'Non-fraudulent']

# total_spend
fig, ax = plt.subplots()
ax.set_title('Boxplot of total_spend')
ax.boxplot(total_spend, vert=False)
ax.set_yticklabels(names)
plt.show()

# doctor_visits
fig, ax = plt.subplots()
ax.set_title('Boxplot of doctor_visits')
ax.boxplot(doctor_visits, vert=False)
ax.set_yticklabels(names)
plt.show()

# num_claims
fig, ax = plt.subplots()
ax.set_title('Boxplot of num_claims')
ax.boxplot(num_claims, vert=False)
ax.set_yticklabels(names)
plt.show()

# member_duration
fig, ax = plt.subplots()
ax.set_title('Boxplot of member_duration')
ax.boxplot(member_duration, vert=False)
ax.set_yticklabels(names)
plt.show()

# optom_presc
fig, ax = plt.subplots()
ax.set_title('Boxplot of optom_presc')
ax.boxplot(optom_presc, vert=False)
ax.set_yticklabels(names)
plt.show()

# num_members
fig, ax = plt.subplots()
ax.set_title('Boxplot of num_members')
ax.boxplot(num_members, vert=False)
ax.set_yticklabels(names)
plt.show()

# c) Orthonormalization of interval variables

y = df['FRAUD']
x = np.matrix([df['TOTAL_SPEND'], df['DOCTOR_VISITS'], df['NUM_CLAIMS'], df['MEMBER_DURATION'], df['OPTOM_PRESC'], df['NUM_MEMBERS']])
x = x.transpose()

print("Input Matrix = \n", x)
print("Number of Dimensions = ", x.ndim)
print("Number of Rows = ", np.size(x,0))
print("Number of Columns = ", np.size(x,1))

xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)))
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx = transf_x.transpose() * transf_x
print("Expect an Identity Matrix = \n", xtx)


# d) Nearest Neighbors

# NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5, algorithm = 'brute' , metric = 'euclidean').fit(transf_x)
distances, indices = nbrs.kneighbors(transf_x)
print("distances = "+str(distances))
print("indices = "+str(indices))

# KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(transf_x, y)
score = neigh.score(transf_x, y)
print("Score = "+str(score))

# e) find 5 neighbors

input_values = [7500, 15, 3, 127, 2, 2]
input_values_transf = input_values * transf
print ("Input variable = "+str(input_values_transf))

myNeighbors = nbrs.kneighbors(input_values_transf, return_distance = False)
print("My Neighbors = \n", myNeighbors)

table_neigh = df.iloc[myNeighbors[0]]
print(str(table_neigh))

# f) prediction

pred_y = neigh.predict(input_values_transf)
print()
print("Prediction : "+str(pred_y))

