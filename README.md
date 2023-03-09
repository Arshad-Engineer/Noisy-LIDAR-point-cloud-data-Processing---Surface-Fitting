# Noisy-LIDAR-point-cloud-data-Processing---Surface-Fitting
# ENPM673 – Perception for Autonomous Robots

# Project 1
Given are two csv files, pc1.csv and pc2.csv, which contain noisy LIDAR point cloud data in the form
of (x, y, z) coordinates of the ground plane.
+ 1. Using pc1.csv:
a. Compute the covariance matrix. [15]
b. Assuming that the ground plane is flat, use the covariance matrix to compute the
magnitude and direction of the surface normal. [15]
+ 2. In this question, you will be required to implement various estimation algorithms such as
Standard Least Squares, Total Least Squares and RANSAC.
a. Using pc1.csv and pc2, fit a surface to the data using the standard least square
method and the total least square method. Plot the results (the surface) for each
method and explain your interpretation of the results. [20]
b. Additionally, fit a surface to the data using RANSAC. You will need to write RANSAC
code from scratch. Briefly explain all the steps of your solution, and the parameters
used. Plot the output surface on the same graph as the data.

## A. File Structure

This projects consists of the following code files
+ Problem #1:
    1. redball_trajectory.py
+ Problem #2:
    1. covariance_surf_normal.py
    2. lstq.py
    3. tsl.py
    4. ransac.py

## B. Modification to the given dataset:
- Add the header to the 3 given columns as 'x','y','z' respectively

## C. Dependancies

+ Ensure the following depenancies are installed
    ```
    pip install pandas
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install opencv-python
    ```

+ Ensure that the above programs are downloaded into the same folder containing 
'pc1.csv' and 'pc2.csv' files

## D. Running the Program

+ Run the programs individually to check the outputs.

+ For lstq.py, tsl.py, ransac.py, you may need to change the file name for two different datasets.
    ```
    python3 <file_name>
    ```
## E. Results
+ On running each of the proframs, the output either pops out a plot or a video in individual window. The outputs can be correlated with the outputs shown in the report.
