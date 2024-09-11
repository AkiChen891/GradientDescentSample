## Gradient Descent Algorithm Sample (one feature)
This project shows the application of Batch Gradient Descent algorithm to fit data.

### dataset_1.csv
This file includes 150 single eigenvalue sample data. The first column is the normalized feature ($x_0$) and the second column is the target value ($h_{\theta}(x)$).

### The purpose of the algorithm
Use gradient descent to implement linear regression to minimize the cost function. From a geometric point of view, it is intuitive to use a single feature function to fit all the data in the scatter plot.

### Hypothesis Function and Cost Function
Hypothesis Function:
$$
h_{\theta}(x)=\theta_0 x_0
$$

Cost Function: 
$$
J(\theta_{0})=\frac{1}{2m} \Sigma (h_{\theta}(x^{(i)})-y^{(i)})^2
$$

In order to make the model's fitting ability more flexible, additional degrees of freedom are introduced. The bias term is usually used as a constant term in the mathematical expression of the model, which helps the model better represent the data characteristics.

The bias term here is denoted as $\theta_1$. Thus hypothesis function can be written as:
$$
h_{\theta}(x)=\theta_0 x_0 + \theta_1
$$

The weight matrix thus includes two elements, which are $\theta_0$ and $\theta_1$. When initializing the matrix, we have mannually set the weight matrix to [0,0].

## Gradient Descent Algorithm Sample (Three features)
Basically same as one feature project. To be continud for more info.