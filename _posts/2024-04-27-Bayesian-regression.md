---
title: Bayesian regression
layout: collection
description: Derivation and usage of Bayesian regression
tag: machine_learning 
mathjax: true
toc: true
---


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from scipy.stats import multivariate_normal
import seaborn as sns 
sns.set_style()
import matplotlib.pyplot as plt

from IPython.display import display, clear_output
```

## General

In bayesian regression, instead of a definit point estimate $y$, we give it a probability distribution, where the mean is the point estimate, but with an extra noise. So our prediction of $y$ is a gaussian distribution signifying our uncertainty of predicting $y$ given the already observed data.

### Setup

Given a datapoint $\mathbf{x} \in \mathbb{R}$, we want to map it to a real valued output $y_i$, where $y_i$ comes from a distribution.

$$
p(y| \mathbf{x}, \mathbf{w}, \sigma^2) = \mathcal{N}(y| \mathbf{w}^T \mathbf{x}, \sigma^2)
$$

We want to find a $\mathbf{w}$ which maximizes the probability of $y$ given $\mathbf{x}$. If we have our data $\mathcal{D} = \{ (\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n) \}$ the likelihood is given by:

$$
L(\mathbf{w}, \mathcal{D}) = \prod_{i=1}^n \mathcal{N}(y| \mathbf{w}^T \mathbf{x}_i, \sigma^2) \propto \exp \left(- \frac{1}{2 \sigma^2} ( y_i - \mathbf{w}^T \mathbf{x}_i)^2 \right)
$$

We keep all our possible $\mathbf{w}$, not only the one that maximizes $y_i$ like in the frequentist view. We then weigh these $\mathbf{w}$ by their posterior probability.

$$
p(y| \mathbf{x}, \mathcal{D}, \sigma^2) = \int p(y | \mathbf{x}, \mathbf{w}, \sigma^2) p(\mathbf{w} | \mathcal{D}, \sigma^2) d\mathbf{w}
$$

This integral can be seen as the averaging over all the models, multiplying the probability of $y$ given the probability of the model $\mathbf{w}$.

#### Posterior

For the posterior of $\mathbf{w}$ we specify our prior belief as:

$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \tau^{2} \mathbf{I})
$$


```python
# define a grid for x and y values
x = np.linspace(-10, 10, 500)  # generate 500 points between -10 and 10 for x
y = np.linspace(-10, 10, 500)  # generate 500 points between -10 and 10 for y
X, Y = np.meshgrid(x, y)  # create a grid for (x,y) pairs

# create an empty array of the same shape as X to hold the (x, y) coordinates
pos = np.empty(X.shape + (2,))

# fill the pos array with the x and y coordinates
pos[:, :, 0] = X  
pos[:, :, 1] = Y  

# create a multivariate normal distribution using the defined parameters
rv = multivariate_normal(np.zeros(2,), np.eye(2))

# create a new figure for 3D plot
fig = plt.figure(figsize=(10, 20))

# add a 3D subplot to the figure
ax = fig.add_subplot(projection='3d')

# create a 3D surface plot of the multivariate normal distribution
ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)

# set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# display the 3D plot
plt.show();
```


    
![png](/images/2024-04-27-Bayesian-regression_files/2024-04-27-Bayesian-regression_3_0.png)
    


The posterior is then given by the bayes theorem:

$$
p(\mathbf{w} | \mathcal{D}, \sigma^2) \propto L(\mathbf{w}; \mathcal{D}) p(\mathbf{w})
$$

Where the posterior again is a gaussian distribution:

$$
p(\mathbf{w} | \mathcal{D}, \sigma^2) = \mathcal{N}(\mathbf{w} | \mathbf{w}_n, \mathbf{V}_n) = 
\mathcal{N} \left( \mathbf{w} | (X^T X + \frac{\sigma^2}{\tau^2} I)^{-1} X^T \mathbf{y}, \sigma^2 (X^T X + \frac{\sigma^2}{\tau^2} I)^{-1} \right) \\
$$

Proof:
$$
\begin{align*}
p(\mathbf{w} | \mathcal{D}, \sigma^2) & \propto L(\mathbf{w}; \mathcal{D}) p(\mathbf{w}) \\
&= \exp \left(- \frac{1}{2 \sigma^2} ( y_i - \mathbf{w}^T \mathbf{x}_i)^2 \right) \exp \left( -\frac{1}{2 \tau^2} \mathbf{w}^T \mathbf{w}\right) \\
&= \exp \left(- \frac{1}{2 \sigma^2} (y - X \mathbf{w})^T (y - X \mathbf{w}) \right) \exp \left( -\frac{1}{2 \tau^2} \mathbf{w}^T \mathbf{w}\right) \\
&= \exp \left(- \frac{1}{2 \sigma^2} (y - X \mathbf{w})^T (y - X \mathbf{w}) -\frac{1}{2 \tau^2} \mathbf{w}^T \mathbf{w}\right) \\
&\propto \exp \left(- \frac{1}{\sigma^2} ( y^T y - 2 y^T X \mathbf{w} + \mathbf{w}^T X^T X \mathbf{w} ) -\frac{1}{\tau^2} \mathbf{w}^T \mathbf{w}\right) \\
&= \exp \left(- \frac{1}{\sigma^2} y^T y + \frac{1}{\sigma^2} 2 y^T X \mathbf{w} -\frac{1}{\sigma^2} \mathbf{w}^T X^T X \mathbf{w} ) -\frac{1}{\tau^2} \mathbf{w}^T \mathbf{w}\right) \\
&=\exp \left(- \frac{1}{\sigma^2} y^T y + \frac{2}{\sigma^2}  y^T X \mathbf{w} - \mathbf{w}^T \underbrace{( \frac{1}{\sigma^2} X^T X + \frac{1}{\tau^2})}_{\Sigma} \mathbf{w} \right)  \\
\end{align*}
$$



```python
def data_generator(f, x: list, y: list, n: int = 1) -> tuple[np.array, np.array]:
    xi = 4 * (np.random.random() - 0.5)
    yi = np.random.normal(loc=f(xi), scale=1)
    x.append(xi)
    y.append(yi)
    return np.array(x), np.array(y)
```

### Basis expansion


```python
def basis_expansion(basis :tuple, x : np.array) -> np.array:
    array_expanded = np.zeros((len(basis), len(x)))
    for i, f in enumerate(basis):
        for j, point in enumerate(x):
            array_expanded[i, j] = f(point)
    return array_expanded.T
            
```

### Posterior distribution


```python
def posterior_distribution(X: np.array, y: np.array, sigma: float, tau: float) -> tuple[np.array, np.array]:
    n, m = X.shape
    #print(X.shape, y.shape, X, y, X.T @ X)
    beta = np.linalg.inv(X.T @ X + (sigma / tau)**2 * np.eye(m)) @ X.T @ y
    covariance = np.linalg.inv((1 / sigma**2) * X.T @ X + (1 / tau**2) * np.eye(m))
    return beta, covariance
```

### Posterior predictive distribution 


```python
def posterior_predictive_distribution(X: np.array, beta: np.array, sigma, covariance: np.array):
    
    pass
```

### BIC score


```python
def bic_score(mean: float, var: float) -> float:
  n = len(x)
  for d in range(1, 20):
    break
      
```

### Implementation


```python
x = []
y = []
f = lambda x : x**3 - x + 1
basis = (lambda x: x**0, lambda x: x**1, lambda x: x**2, lambda x: x**3)

plt.figure(figsize=(10, 6))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian linear regression')
x_points = np.linspace(-1.5, 1.5)
for i in range(20):  
    x_array, y_array = data_generator(f, x, y)
    
    X = basis_expansion(basis, x_array)
    beta, covariance = posterior_distribution(X, y_array, 1, 1)
    y_pred = basis_expansion(basis, x_points) @ beta.T
    
    plt.plot(x_points, y_pred, color='red', alpha=0.1)
    sns.scatterplot(x=x, y=y)
    #sns.lineplot(np.linspace(-2, 2, 100), f(np.linspace(-2, 2, 100)), color='red')
    
    plt.legend(['Random points', 'Original function'])
    plt.show()
    clear_output(wait=True)
    plt.pause(1)
```


```python

```
