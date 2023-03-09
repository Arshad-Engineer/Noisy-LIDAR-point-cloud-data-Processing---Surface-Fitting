# Noisy-LIDAR-Point-Cloud-Data-Processing & Surface-Fit Estimation using Least Squares, Total Least Squares, RANSAC
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
used. Plot the output surface on the same graph as the data.Discuss which graph
fitting method would be a better choice of outlier rejection. [20]

## A. File Structure

This projects consists of the following code files

+ Problem #2:
    1. covariance_surf_normal.py
    2. lstq.py
    3. tsl.py
    4. ransac.py

## B. Modification to the given dataset:
- na

## C. Dependancies

+ Ensure the following depenancies are installed
    ```
    pip install pandas
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install opencv-python
    ```

## D. Running the Program

+ Run the programs individually to check the outputs.

+ For lstq.py, tsl.py, ransac.py, you may need to change the file name for two different datasets.
    ```
    python3 <file_name>
    ```
## E. Results
+ On running each of the proframs, the output either pops out a plot or a video in individual window. The outputs can be correlated with the outputs shown in the report.
+ Plot: Least Squares method: Dataset pc1.csv
![image](https://user-images.githubusercontent.com/112987383/223980842-a1ba0ae6-5633-4331-8cf3-70fda3dde5e7.png)
+ Plot: Total Least square Method: Dataset pc1.csv
![image](https://user-images.githubusercontent.com/112987383/223981206-dfc00c67-4b25-4c5f-8171-4669ea8e96b9.png)
+ Plot: RANSAC Method: Dataset pc1.csv
![image](https://user-images.githubusercontent.com/112987383/223981114-db879f66-9b8c-414e-b747-dfa85df96ce6.png)


