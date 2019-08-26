# import All requried libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

# import  Dataset:
data = pd.read_csv('Advertising.csv')

# droping the unusefull column:
data.drop(['Unnamed: 0'] ,axis =1, inplace = True)

# making the dataset in the form of dependent and independent :
X = data['TV']
y = data['sales']

# ploting the sactter plot of the dependent nad indepednet dataset:
plt.scatter(X ,y)

# Now make the function for Best_Fit_Slop
# for  linear regression line:

def best_fit_slop(X ,y):
    m = ((mean(X) * mean(y)) - mean(X * y)) / ((mean(X) * mean(X)) - mean(X * X))
    return m
m = best_fit_slop(X , y)

# Making the funtion for best_fit_intercepts:
def best_fit_intercept(X , y):
    b = mean(y) - m * mean(X)
    return b
b = best_fit_intercept(X ,y)

# Now making the combine equation of best_fit_slop and best_fit_intercets:
def best_fit_slop_and_intercept(X,y):
    m = ((mean(X) * mean(y)) - mean(X*y)) / ((mean(X) * mean(X)) - mean(X * X))  
    b =mean(y) - m * mean(X)  
    return m , b

# HOW TO CALCULATE R- SQUARED VALUE:
# Step -- 1:
# Make the function to calculate the Squared_error:
def square_error(y_orig ,y_line):
    return sum((y_orig - y_line)**2)

# Step -- 2:
# Make the function to Calculate R_Squared:
def cofficient_of_determination(y_orig , y_line):
    y_mean_line =[mean(y_orig) for y in y_orig]
    squared_error_regr = square_error(y_orig , y_line)
    squared_error_y_mean = square_error(y_orig,y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
r_sqaured = cofficient_of_determination(y , regression_line)

m , b =best_fit_slop_and_intercept(X , y)

# Now make a Regression_line for look how much fit on data
regression_line = [(m*x) + b for x in X]
plt.scatter(X,y , color ='blue')
plt.plot(X , regression_line,color = 'red')
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()

# After the  calculating the R_squared:
# Now we want to predict of a point:
predict_X = 202.5
predict_y = (m * predict_X) + b
plt.scatter(X,y,color='blue')
plt.scatter(predict_X ,predict_y ,color ='red')
plt.plot(X ,regression_line, color ='yellow')
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()
print(r_sqaured)

# END OF CODE:





