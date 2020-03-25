import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv('NormalSample.csv')
N = df.shape[0]
x = df['x']

# sort x
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x
x = selection_sort(np.array(x))
print (x)

# a) 5 number summary and 1.5 IQR whiskers

min = np.min(x)
print("min = "+str(min))
Q1 = np.percentile(x, 25)
print("Q1 = "+str(Q1))
median = np.median(x)
print("median = "+str(median))
Q3 = np.percentile(x, 75)
print("Q3 = "+str(Q3))
max = np.max(x)
print("max = "+str(max))

IQR = scipy.stats.iqr(x)
print("IQR = "+str(IQR))

low_whisk = Q1 - 1.5*IQR
up_whisk = Q3 + 1.5*IQR
print("lower whisker  = "+str(low_whisk))
print("upper whisker  = "+str(up_whisk))

# b)  5 number summary and 1.5 IQR whiskers for each group

# group 0

df_0 = df[df['group'] == 0]
print(df_0)

x_0 = df_0['x']
x_0 = selection_sort(np.array(x_0))
print (x_0)

min_0 = np.min(x_0)
print("Group 0 min = "+str(min_0))
Q1_0 = np.percentile(x_0, 25)
print("Group 0 Q1 = "+str(Q1_0))
median_0 = np.median(x_0)
print("Group 0 median = "+str(median_0))
Q3_0 = np.percentile(x_0, 75)
print("Group 0 Q3 = "+str(Q3_0))
max_0 = np.max(x_0)
print("Group 0 max = "+str(max_0))

IQR_0 = scipy.stats.iqr(x_0)
print("Group 0 IQR = "+str(IQR_0))

low_whisk_0 = Q1_0 - 1.5*IQR_0
up_whisk_0 = Q3_0 + 1.5*IQR_0
print("Group 0 lower whisker  = "+str(low_whisk_0))
print("Group 0 upper whisker  = "+str(up_whisk_0))


# group 1

df_1 = df[df['group'] == 1]
print(df_1)

x_1 = df_1['x']
x_1 = selection_sort(np.array(x_1))
print (x_1)

min_1 = np.min(x_1)
print("Group 1 min = "+str(min_1))
Q1_1 = np.percentile(x_1, 25)
print("Group 1 Q1 = "+str(Q1_1))
median_1 = np.median(x_1)
print("Group 1 median = "+str(median_1))
Q3_1 = np.percentile(x_1, 75)
print("Group 1 Q3 = "+str(Q3_1))
max_1 = np.max(x_1)
print("Group 1 max = "+str(max_1))

IQR_1 = scipy.stats.iqr(x_1)
print("Group 1 IQR = "+str(IQR_1))

low_whisk_1 = Q1_1 - 1.5*IQR_1
up_whisk_1 = Q3_1 + 1.5*IQR_1
print("Group 1 lower whisker  = "+str(low_whisk_1))
print("Group 1 upper whisker  = "+str(up_whisk_1))

# c) Boxplot of x

fig1, ax1 = plt.subplots()
ax1.set_title('Boxplot of x')
ax1.boxplot(x)
plt.show()

# d) Boxplot of x of the entire data and of each group and outliers
data = [x, x_0, x_1]
fig7, ax7 = plt.subplots()
ax7.set_title('Boxplot of entire data, group 1 and group 2')
ax7.boxplot(data)

plt.show()

# outliers

a = np.array(x[x<low_whisk])
b = np.array(x[x>up_whisk])
out_x = np.concatenate((a,b), axis=0)
print("Outliers of x for the entire data = "+str(out_x))

a = np.array(x_0[x_0<low_whisk_0])
b = np.array(x_0[x_0>up_whisk_0])
out_x_0 = np.concatenate((a,b), axis=0)
print("Outliers of x for Group 0 = "+str(out_x_0))

a = np.array(x_1[x_1<low_whisk_1])
b = np.array(x_1[x_1>up_whisk_1])
out_x_1 = np.concatenate((a,b), axis=0)
print("Outliers of x for Group 1 = "+str(out_x_1))