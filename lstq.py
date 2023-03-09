# %% [markdown]
# # Least Squares Solution:

# %% [markdown]
# <<<<< **Author: Arshad Shaik, UID:118438832; ENPM-673 Perception for Robotics; part of Project-1**>>>>
# 
# The following program uses pc1.csv and pc2.csv, fits a surface to the data using the least square method. Subsequently, the results (the surface) are plotted to demonstrate the fit.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
#%matplotlib inline

# %%
df = pd.read_csv('pc2.csv') # change the dataset as needed
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
data = np.hstack((XXX,YYY,ZZZ))

# %%
X,Y = np.meshgrid(np.arange(-12.0, 12.0, 0.5), np.arange(-12.0, 12.0, 0.5))

# %%
print(np.shape(X))

# %%
print(np.shape(Y))

# %%
sum_x = 0.0

for i in range(0,len(XXX)):
    sum_x += XXX[i]

print(sum_x)

# %%
sum_y = 0.0

for i in range(0,len(YYY)):
    sum_y += YYY[i]

print(sum_y)

# %%
sum_z = 0.0

for i in range(0,len(ZZZ)):
    sum_z += ZZZ[i]

print(sum_z)

# %%
sum_x2 = 0.0

for i in range(0,len(XXX)):
    sum_x2 += XXX[i]*XXX[i]

print(sum_x2)

# %%
sum_y2 = 0.0

for i in range(0,len(YYY)):
    sum_y2 += YYY[i]*YYY[i]

print(sum_y2)

# %%
sum_z2 = 0.0

for i in range(0,len(ZZZ)):
    sum_z2 += ZZZ[i]*ZZZ[i]

print(sum_z2)

# %%
sum_xy = 0.0

for i in range(0,len(XXX)):
    sum_xy += XXX[i]*YYY[i]

print(sum_xy)

# %%
sum_xz = 0.0

for i in range(0,len(XXX)):
    sum_xz += XXX[i]*ZZZ[i]

print(sum_xz)

# %%
sum_yz = 0.0

for i in range(0,len(YYY)):
    sum_yz += YYY[i]*ZZZ[i]

print(sum_yz)

# %%
AA = np.array([[sum_x2[0], sum_xy[0], sum_x[0]], [sum_xy[0], sum_y2[0], sum_y[0]], [sum_x[0], sum_y[0], len(XXX)]], dtype=np.float64)
BB = np.array([sum_xz, sum_yz, sum_z])
Q = np.linalg.inv(AA).dot(BB)

print(Q)

# %%
  # evaluate it on grid
Z = Q[0]*X + Q[1]*Y + Q[2]

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



