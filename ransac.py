import matplotlib.pyplot as plt
import numpy as np
import math
import csv


#Function to fit a plane to the inliers:
def fitting_plane(start_inliers): 
    A = start_inliers[0] #first x,y,z point
    B = start_inliers[1] #second x,y,z point
    C = start_inliers[2] #third x,y,z point

    AB = B - A  #vector from A to B
    AC = C - A  #vector from A to C
    n = np.cross(AB, AC) #cross product of AB and AC gives 'n' (gives a,b,c of plane equation)
    d = - (n[0]*A[0] + n[1]*A[1] + n[2]*A[2]) #'d' in the plane equation 

    params = np.append(n,d) #plane equation parameters
    return params


#Function to calculate distance of a point from a plane:
def point_2_plane(plane_params, point): 
    plane_eq = math.fabs(plane_params[0]*point[0]+plane_params[1]*point[1]+plane_params[2]*point[2]+plane_params[3]) #plane equation
    d_normal = np.linalg.norm(plane_params[:3]) #normal distance
    return plane_eq/d_normal #distance of point from plane
    

#Function to check all other points in dataset whether inlier/outlier: 
def alsoInliers(data_in, plane_params, threshold): 
    flag = 0 
    alsoinliers = np.array([]) #array to store inliers
    for i in range(0, data_in.shape[0]): #loop through all points in dataset
        dis = point_2_plane(plane_params, data_in[i])
        if (dis < threshold): #if distance is less than threshold, it is an inlier
            if flag == 0:
                alsoinliers = i
            else:
                alsoinliers = np.append(alsoinliers, i)
            flag = flag + 1

    return alsoinliers


#Read data from csv file:
with open('pc1.csv') as file: #Importing data from csv file
        csv_reader = csv.reader(file)
        i = 0    
        for row in csv_reader:
            if i==0:
                row = [float(x) for x in row[0:3]] #convert string to float
                row1 = np.array(row[0:3]) #create array and store first row in 'row1'
            elif i>0:
                row = [float(x) for x in row[0:3]]
                row1 = np.row_stack((row1, row)) #stack all rows in 'row1' one below the other
            i=i+1

#Normalize data to avoid numerical errors and easier to fit a plane
normalize_1 = (row1[:,0] - np.mean(row1[:,0]))/np.std(row1[:,0]) #normalize x by subtracting mean and dividing by standard deviation
normalize_2 = (row1[:,1] - np.mean(row1[:,1]))/np.std(row1[:,1]) #normalize y by subtracting mean and dividing by standard deviation 
normalize_3 = (row1[:,2] - np.mean(row1[:,2]))/np.std(row1[:,2]) #normalize z by subtracting mean and dividing by standard deviation

data_in = normalize_1 #create a new array to store normalized data

data_in = np.column_stack((data_in, normalize_2)) #stack all normalized data in 'data_in' one after the other
data_in = np.column_stack((data_in, normalize_3)) #stack all normalized data in 'data_in' one after the other


#Initial Parameters: 
max_iterations = 1000 #maximum number of iterations for ransac
threshold = 0.2 #threshold for distance of point from plane - if less than threshold, it is an inlier
fit_prob = 0.99 #probability that all points are inliers
max_count = 0 #maximum number of inliers
inliers = data_in #array to store inliers (final) after ransac

#RANSAC Algorithm:
for i in range(0, max_iterations):
    choice = np.random.choice(data_in.shape[0],3,replace=False) #randomly select 3 points from dataset
    hypo_inliers = data_in[choice] #store the 3 points in 'hypo_inliers'
    plane_params = fitting_plane(hypo_inliers) #fit a plane to the 3 points

    temp_inliers = alsoInliers(data_in, plane_params, threshold) #check all other points in dataset whether inlier/outlier

    if temp_inliers.shape[0] > max_count: #if number of inliers is greater than max_count, update max_count and inliers
        max_count = temp_inliers.shape[0]
        inliers = temp_inliers
        best_plane = plane_params #store the best plane parameters
    
    if (inliers.shape[0]/data_in.shape[0] > fit_prob): #if number of inliers vs total data points is greater than fit_prob, break
        break

#Storing the inliers and outliers in 'res' array:
idx = 0
for i in range(0, data_in.shape[0]): #loop through all points in dataset
    row = data_in[i] #store each point in 'row'
   
    if idx < inliers.shape[0] and inliers[idx] == i: #if the point is an inlier
        idx = idx + 1 
        row = np.append(row, 1) #append 1 to the end of 'row' to indicate it is an inlier
    else:
        row = np.append(row, 0) #append 0 to the end of 'row' to indicate it is an outlier
    
    if i == 0: #store the first row in 'res' and then stack all other rows below it
        res = row
    else:
        res = np.row_stack((res, row))


#Plotting the plane and points:   
x = np.linspace(res[:,0].min(), res[:,0].max(), 30) #create a grid of points to plot the plane
y = np.linspace(res[:,1].min(), res[:,1].max(), 30)  
x1, y1 = np.meshgrid(x, y) #create a meshgrid of points to plot the plane
z = (-best_plane[0] * x1 - best_plane[1] * y1 - best_plane[3]) * 1. /best_plane[2] #plane equation

plot1 = plt.figure()
plot1 = plt.axes(projection='3d')
plot1.set_title("pc1.csv - RANSAC")
plot1.set_xlabel("X-Axis - Measurement")
plot1.set_ylabel("Y-Axis - Measurement") 
plot1.set_zlabel("Z-Axis - Measurement") 

plot1.plot_surface(x1, y1, z, alpha=0.4) #plot the plane - transparent
plot1.scatter(res[:,0], res[:,1], res[:,2], marker='.', c = res[:,3], cmap = 'PiYG') #plot the data points - green if inlier, violet if outlier

plt.show()  