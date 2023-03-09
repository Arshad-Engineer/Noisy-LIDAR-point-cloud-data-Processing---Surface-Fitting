# Noisy-LIDAR-point-cloud-data-Processing---Surface-Fitting
# ENPM673 – Perception for Autonomous Robots

# Project 1

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
