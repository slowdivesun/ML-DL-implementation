# ML-DL-implementation
An implementation of ML and DL algorithms from scratch in python using nothing but numpy.
# Linear_example.py
This is an explanation of the linar_example.py module
From line first through fourth, we import all the relevant functions.

The read_data function allows us to get the (x, y) pairs of values from the raw data in the txt file.

After that, we initialise an instance of the LinearRegression class and store it in a variable named linear_model. 

# MeanSquaredError
There are two static methods defined in this class. Follow this [link to read about static and class methods](https://www.youtube.com/watch?v=rq8cL2XMM5M0)

- Loss Function
It receives 3 parameters `X, Y and W`, which are all matrices.
Then it stores the total number of test values in the variable M. `X.shape[0]` gives the first element of the tuple X.shape, which is ultimately the total number of columns in the 1D matrix X.
The function then takes the dot product of X and W, takes the transpose(.T) of the resulting matrix, then subtracts Y from it. The matrix obtained gives the errors for each sample point. 
Each of the errors are then squared and added. After this the result is divided by 2*M.
This final value is returned as loss, which in simple words is 'total error'.

# misc_utils.py 
- read_data

  A file name, along with the extension (.txt in out example) is provided to the function
The genfromtext function in numpy enables us to transfer the data to well defined matrices
The last column of data is the Y value, that is the output of the overall mathematical fuction that we're dealing with. For example, if we have 3 variables, 2 independent and 1 dependent in our data sample. In this case the first 2 columns are for X1 and X2 while the last one is for Y. The data in the txt file should be seperated using a delimiter, commonly `,`
