import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gd = pd.read_csv('dataset_1.csv')


# X = feature matrix, column = feature, row = sample
# y = target value
# X * theta.T : Predicted values ​​from the linear regression model
# np.power(): Calculates the power of array elements, e.g. np.power(x,2)=Calculate the square of all elements in the matrix x
# X.shape[0]: row amounts (sample size) for X. E.g. shape[1] returns columns
# np.sum(inner): Find the sum of the squared errors of all samples
def computecost(X,y,theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner)/(2*X.shape[0])

# Insert a column of all 1s as a bias term
gd.insert(0,'ones',1)

# To initialize the variables, set X to the training set data and y to the target variable
# Typically, we consider all columns except the last one as features, that is, assigned to X; and the last column as the target variable or label, that is, y.
cols = gd.shape[1] # column size of gd
X=gd.iloc[ : ,        : cols-1] # Select first column(Feature)
y=gd.iloc[ : , cols-1 : cols] # Select last column(Target)


# Transfer X,y,theta to matrix
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))

# X:Feature matrix(size=m*n) ; y:Target(size=m*1) ; theta:Weights(size=1*n)
# alpha:Learning rate ; iters:Iterations，indicates how many times to perform gradient descent updates
def batch_gradientdescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape)) # Create a zero matrix of the same shape as theta to store the updated weights after each iteration.
    # Flatten theta into one dimension and get its number of columns, indicating the number of features. 
    # ravel() flattens a multidimensional array into one dimension, and shape[1] returns the number of columns (that is, the number of features, including the weight of the bias term)
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters) # Store the cost (loss) of each iteration
    # Main loop for gradient descent iteration
    for i in range(iters):
        error = (X*theta.T) - y # Calculate error, size=m*1
        # Inner loop: Update weights
        for j in range(parameters): # Update the weight theta[j] corresponding to each feature one by one.
            term = np.multiply(error,X[ : , j]) # For each feature, calculate the product of the error and the feature value
            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term)) # Update weights, alpha / len(X) is the ratio of learning rate to number of samples, and np.sum(term) is the gradient.
        theta = temp # Assign temporary weights to the weight vector
        cost[i] = computecost(X,y,theta) # Calculate the cost (loss) under the current weights and store it in the cost array.
    return theta,cost

alpha=0.02 # Learning rate
iters=250 # Iterations

# Execute gradient descent
g,cost=batch_gradientdescent(X,y,theta,alpha,iters)

# Print final weights
print("Final theta values:\n", g)
print("Final cost:", computecost(X, y, g))

gd.plot(kind='scatter',x='Feature1',y='Target',figsize=(12,8))
plt.xlabel('Feature',fontsize=12)
plt.ylabel('Target',rotation=90,fontsize=12)
plt.title('Scatter Plot',fontsize=16)
x_vals = np.linspace(gd['Feature1'].min(), gd['Feature1'].max(), 100)
y_vals = g[0, 0] + g[0, 1] * x_vals
plt.plot(x_vals, y_vals, color='red', label='Fitted Line')
plt.legend()
plt.show()

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations',fontsize=12)
ax.set_ylabel('cost',rotation=90,fontsize=12)
ax.set_title('Error-Iterations',fontsize=15)
plt.show()
