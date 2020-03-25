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

# a) bin-width
IQR = scipy.stats.iqr(x)
h = 2*IQR*(N**(-1/3))
print("a) bin-width h = "+ str(h))

# b) Min and max vals of x
min = min(x)
max= max(x)
print("b) Min value = "+str(min)+" Max value = "+str(max))

# c) Min int and max int vals of x
min_int = int(min)
max_int = math.ceil(max)
print("c) Min int value = "+str(min_int)+" Max int value = "+str(max_int))

def den_estimate(mid_points,x,N,h):
    p_vect = []
    for m in mid_points:
        res = [i - m for i in x]
        u = [i / h for i in res]
        w = []
        for i in range(len(u)):
            if u[i] > -1 / 2 and u[i] <= 1 / 2:
                w.append(1)
            else:
                w.append(0)
        p = sum(w) / (N * h)
        p_vect.append(p)

    return p_vect

# d) Density estimator and histogram (h = 0.25)

h = 0.25
# computation of mid points
mid_points = []
mid_points.append(min_int + h/2)

while mid_points[-1]+h <= max_int:
    mid_points.append(mid_points[-1]+h)

print("mid points = "+str(mid_points))

# computation of density estimator
p = den_estimate(mid_points,x,N,h)
print("d) Density estimator = "+ str(p))

# histogram
figure, ax = plt.subplots()
ax.bar(x=mid_points,height=p,width=h)
figure.savefig('histogram_h={}.png'.format(h))
plt.ylabel('Probability')
plt.xlabel('x')
plt.title('Histogram of x with h = {}'.format(h))
plt.show()

# e) Density estimator and histogram (h = 0.5)

h = 0.5
# computation of mid points
mid_points = []
mid_points.append(min_int + h/2)

while mid_points[-1]+h <= max_int:
    mid_points.append(mid_points[-1]+h)

print("mid points = "+str(mid_points))

# computation of density estimator
p = den_estimate(mid_points,x,N,h)
print("e) Density estimator = "+ str(p))

# histogram
figure, ax = plt.subplots()
ax.bar(x=mid_points,height=p,width=h)
figure.savefig('histogram_h={}.png'.format(h))
plt.ylabel('Probability')
plt.xlabel('x')
plt.title('Histogram of x with h = {}'.format(h))
plt.show()

# f) Density estimator and histogram (h = 1)

h = 1
# computation of mid points
mid_points = []
mid_points.append(min_int + h/2)

while mid_points[-1]+h <= max_int:
    mid_points.append(mid_points[-1]+h)

print("mid points = "+str(mid_points))

# computation of density estimator
p = den_estimate(mid_points,x,N,h)
print("f) Density estimator = "+ str(p))

# histogram
figure, ax = plt.subplots()
ax.bar(x=mid_points,height=p,width=h)
figure.savefig('histogram_h={}.png'.format(h))
plt.ylabel('Probability')
plt.xlabel('x')
plt.title('Histogram of x with h = {}'.format(h))
plt.show()

# g) Density estimator and histogram (h = 2)

h = 2
# computation of mid points
mid_points = []
mid_points.append(min_int + h/2)

while mid_points[-1]+h <= max_int:
    mid_points.append(mid_points[-1]+h)

print("mid points = "+str(mid_points))

# computation of density estimator
p = den_estimate(mid_points,x,N,h)
print("g) Density estimator = "+ str(p))

# histogram
figure, ax = plt.subplots()
ax.bar(x=mid_points,height=p,width=h)
figure.savefig('histogram_h={}.png'.format(h))
plt.ylabel('Probability')
plt.xlabel('x')
plt.title('Histogram of x with h = {}'.format(h))
plt.show()