# %% [markdown]
# # Total Least Square Solution:

# %% [markdown]
# <<<<< Author: Arshad Shaik, UID:118438832; ENPM-673 Perception for Robotics; part of Project-1>>>>
# 
# The following program uses pc1.csv and pc2.csv, fits a surface to the data using the total least square method. Subsequently, the results (the surface) are plotted to demonstrate the fit.

# %%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.linalg
#%matplotlib inline

# %%
df = pd.read_csv('pc1.csv') # change the dataset as needed
print(df)

# %%
XX = df['x'].values

# %%
XXX = np.array([XX]).T

# %%
np.shape(XXX)

# %%
YY = df['y'].values
YYY = np.array([YY]).T
ZZ = df['z'].values
ZZZ = np.array([ZZ]).T

# %%
np.shape(YYY)

# %%
data = np.hstack((XXX,YYY,ZZZ))

# %%
X,Y = np.meshgrid(np.arange(-12.0, 12.0, 0.5), np.arange(-12.0, 12.0, 0.5))

# %%
A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0]),data[:,2] ]
A

# %%
print(np.shape(A))

# %%
# Reference: https://en.wikipedia.org/wiki/Singular_value_decomposition
w, v = LA.eig(A.T @ A)

# %%
w

# %%
sig = np.diag(np.sqrt(np.abs(w)))
sig

# %%
v

# %%
np.shape(v)

# %%
vt = v.T

# %%
U = A @ v @ np.linalg.inv(sig)

# %%
U

# %%
n = np.array(A).shape[1] - 1
n


# %%
Vxy = v[:n, n:]
Vxy

# %%
Vyy = v[n:, n:]
Vyy

# %%
a_tls = - Vxy  / Vyy
print("a_tls", a_tls)


# %%
A1 = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]

# %%
# y_tls = (A1+Xt).dot(a_tls) 
# y_tls
Z = a_tls[0]*X + a_tls[1]*Y + a_tls[2]

# %%
# plot points and fitted surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('auto')
ax.axis('tight')
plt.show()

# %%



