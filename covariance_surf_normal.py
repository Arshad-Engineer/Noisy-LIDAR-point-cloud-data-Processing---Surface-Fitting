# %% [markdown]
# # Covariance of the Matrix

# %% [markdown]
# <<<<< Author: Arshad Shaik, UID:118438832; ENPM-673 Perception for Robotics; part of Project-1>>>>
# 
# The following program uses pc1.csv and computes the covariance matrix, and using this matrix, the magnitude and direction of surface normal, are calculated.

# %%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from math import tanh, pi, degrees

# %%
# read the csv file
df = pd.read_csv('pc1.csv')
df.head()

# %%
df.describe()

# %%
df.info()

# %%
x = df['x'].values

# %%
type(x)

# %%
y = df['y'].values

# %%
z = df['z'].values

# %%
x_mean = np.mean(x)
print("x_mean:", x_mean)

# %%
y_mean = np.mean(y)
print("y_mean:", y_mean)

# %%
z_mean = np.mean(z)
print("z_mean:", z_mean)

# %%
cov_xx = np.sum((x - x_mean)**2 / (len(x)-1))

# %%
cov_yy = np.sum((y - y_mean)**2 / (len(x)-1))

# %%
cov_zz = np.sum((z - z_mean)**2 / (len(x)-1))

# %%
cov_xy = np.sum((x - x_mean) * (y - y_mean) / (len(x)-1))

# %%
cov_xz = np.sum((x - x_mean) * (z - z_mean) / (len(x)-1))

# %%
cov_yz = np.sum((y - y_mean) * (z - z_mean) / (len(x)-1))

# %%
cov_mx = np.array([[cov_xx, cov_xy, cov_xz], 
                    [cov_xy, cov_yy, cov_yz], 
                    [cov_xz, cov_yz, cov_zz]])

# %%
print("\n Covariance Matrix: \n", cov_mx)

# %%
eig_val, eig_vec = LA.linalg.eig(cov_mx)
print("\n Eigen Values: ", eig_val)
print("\n Eigen Vectors: ", eig_vec)

# %%
min_eig = np.min(eig_val)
min_eig_idx = np.argmin(eig_val)
print("\n Minimum eigen value: " + str(min_eig) + "\t & its index: " + str(min_eig_idx))

# %%
print("\nMagnitude of Surface Normal: ", np.linalg.norm(eig_vec[:,min_eig_idx]))

# %%
#Calc the angle of the first eigenvector (smallest eigenvalue):
theta = tanh(eig_vec[1,0]/eig_vec[0,0]) 
print("\nAngle between projection of surface normal on xy plane & x-axis: " , round(degrees(theta)), "°")

# %%
# magnitude of projection of surface normal on xy plane
mag = np.sqrt(eig_vec[0,0]**2 + eig_vec[1,0]**2) 
# angle between projection of surface normal on xy plane & surface normal
phi = np.tanh(eig_vec[2,0]/mag) 
print("\nAngle between projection of surface normal on xy plane & surface normal: ", round(degrees(phi)), "°")

# %%
psi = (pi/2) - phi #angle between surface normal & z-axis
print("\nAngle between surface normal & z- axis: ", round(degrees(psi)), "°")

# %%



