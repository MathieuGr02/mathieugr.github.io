---
title: Expectation Maximization
layout: collection
permalink: /Computational-Biology/Expectation-Maximization
collection: Computational-Biology
mathjax: true
toc: true
categories:
  - study
tags:
  - mathematics
  - statistics
---


```python
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

sns.set_theme()
```

### Difference trial and reference sample

Assume we have a cell culture where there exist an unknown underlying distribution with mean $\mu_r$ and variance $\sigma^2$ and we know perturb the system with which we get a second distribution with log values with unknown mean $\mu_t$ and variance $\sigma^2$. 

$$
\begin{align*}
\mathbb{P}(x_1, ... | \mu_r, \sigma) 
&= 
\frac{1}{\sigma^n (2 \pi)^{\frac{n}{2}}} \exp \left(- \frac{n}{2\sigma^2} \left[ (\bar{x} - \mu_r)^2 + var(x)\right] \right)
\\
\mathbb{P}(y_1, ... | \mu_r, \sigma) 
&= 
\frac{1}{\sigma^n (2 \pi)^{\frac{n}{2}}} \exp \left(- \frac{n}{2\sigma^2} \left[ (\bar{y} - \mu_t)^2 + var(y)\right] \right)
\end{align*}
$$

Given these two independent distributions, 
we may now want to calculate true expression values of the cells, i.e. the joint distribution $ \mathbb{P}(\mu_r, \mu_t | \mathcal{D}) $.

$$
\mathbb{P}(\mu_r, \mu_t | \mathcal{D}) 
= 
\int \mathbb{P}(\mu_r, \mu_t, \sigma | \mathcal{D}) d\sigma 
= 
\int \frac{\mathbb{P}(\mathcal{D} | \mu_r, \mu_t, \sigma) \mathbb{P}(\mu_r, \mu_t, \sigma)}{\mathbb{P}(\mathcal{D})} d\sigma
$$

With the assumption of a uniform prior we get

$$
\begin{align*}
    \mathbb{P}(\mu_r, \mu_t | \mathcal{D}) 
    &\propto
    \int \mathbb{P}(\mathcal{D} | \mu_r, \mu_t, \sigma) d\sigma \\
    &\propto
    \int \frac{1}{\sigma^{2n}} \exp \left( - \frac{2}{2\sigma^2} \left[(\bar{x} - \mu_r)^2 + (\bar{y} - \mu_r)^2 + var(x) + var(y) \right] \right) d\sigma
\end{align*}
$$

With the same integration trick as with the student-t distribution we get

$$
\mathbb{P}(\mu_r, \mu_t | \mathcal{D}) \propto \left[ var(x) + var(y) + (\mu_r - \bar{x})^2 + (\mu_t - \bar{y})^2 \right]^{-\frac{2n-1}{2}}
$$

When we now want to look at the means, we can ask for example, wether $\mu_t > \mu_r$ or not. We write $\mu = \frac{\mu_r + \mu_t}{2}$, $\delta = \mu_t - \mu_r$ or $\mu_r = \mu - \frac{\delta}{2}$, $\mu_t = \mu + \frac{\delta}{2} $. Plugging this in we get

$$
\begin{align*}
    \mathbb{P}(\mu_r, \mu_t | \mathcal{D}) 
    &\propto 
    \left[ var(x) + var(y) + (\mu_r - \bar{x})^2 + (\mu_t - \bar{y})^2 \right]^{-\frac{2n-1}{2}} \\
    &=
    \left[ var(x) + var(y) + \frac{\delta^2}{2} + 2 \mu^2 + \bar{x}^2 + \delta \bar{x} - 2\mu \bar{x} + \bar{y}^2 - \delta\bar{y} - 2 \mu \bar{y} \right]^{-\frac{2n-1}{2}} \\
\end{align*}
$$

Looking at the terms with $\mu$

$$
\begin{align*}
    & 2 \mu^2 - 2\mu(\bar{x} + \bar{y}) + \bar{x}^2 + \bar{y}^2 \\
    & 2 \mu^2 - 2\mu(\bar{x} + \bar{y}) + \frac{\bar{x}^2}{2}  + \frac{\bar{y}^2}{2} + \frac{\bar{x}^2}{2} + \frac{\bar{y}^2}{2} \\
    & 2 \left( \mu^2 - \mu(\bar{x} + \bar{y}) + \frac{\bar{x}^2}{4}  + \frac{\bar{y}^2}{4} \right)  + \frac{\bar{x}^2}{2} + \frac{\bar{y}^2}{2} \\
    & 2 \left( \mu^2 - \mu(\bar{x} + \bar{y}) + \frac{\bar{x}^2}{4}  + \frac{\bar{y}^2}{4} - \frac{\bar{x}\bar{y}}{2} \right)  + \frac{\bar{x}^2}{2} + \frac{\bar{y}^2}{2} - \bar{x}\bar{y} \\
    & 2 \left( \mu - \frac{\bar{x} + \bar{y}}{2} \right) + \frac{\bar{x}^2}{2} + \frac{\bar{y}^2}{2} - \bar{x}\bar{y} \\
\end{align*}
$$

Then looking at the terms with $\delta$:

$$
\begin{align*}
    & \frac{\delta^2}{2} - \delta(\bar{y} - \bar{x}) + \frac{\bar{x}^2}{2} + \frac{\bar{y}^2}{2} - \bar{x}\bar{y} \\
    & \frac{1}{2}\left(\delta - (\bar{y} - \bar{x}) \right)^2
\end{align*}
$$

Thus we get

$$
\mathbb{P}(\mu_r, \mu_t | \mathcal{D}) 
\propto
\left[ var(x) + var(y) + 2(\mu - \frac{\bar{x} + \bar{y}}{2})^2 + \frac{1}{2}(\delta - (\bar{y} - \bar{x}))^2 \right]^{-\frac{2n-1}{2}}
$$

To then obtain the marginal for $\delta$ we need to integrate 

$$
\begin{align*}
    \mathbb{P}(\delta | \mathcal{D})  
    &= 
    \int \mathbb{P}(\mu, \delta | D) d\mu \\
    &\propto
    \int \left[ var(x) + var(y) + 2(\mu - \frac{\bar{x} + \bar{y}}{2})^2 + \frac{1}{2}(\delta - (\bar{y} - \bar{x}))^2 \right]^{-\frac{2n-1}{2}} d\mu \\
    &\propto
    \int \left[2(\mu - A)^2 + B \right]^{-\frac{2n-1}{2}} d\mu
\end{align*}
$$

The if we define the substitution $u := \sqrt{\frac{2}{B}(\mu - A)} \Rightarrow \sqrt{\frac{B}{2}} du = d\mu$, we get

$$
\begin{align*}
    &\int \left[2(\mu - A)^2 + B \right]^{-\frac{2n-1}{2}} d\mu \\
    &\int \sqrt{\frac{B}{2}} \left[B u^2 + B \right]^{-\frac{2n-1}{2}} du \\
    &\int \sqrt{\frac{B}{2}} B^{-\frac{2n-1}{2}} \left[u^2 + 1 \right]^{-\frac{2n-1}{2}} du \\
    \sqrt{\frac{1}{2}} B^{-\frac{2n-2}{2}} &\int \left[u^2 + 1 \right]^{-\frac{2n-1}{2}} du \\
    &\propto  B^{-(n-1)}
\end{align*}
$$

Thus in the end we get the marginal distribution, which again is a student-t distribution

$$
\begin{align*}
\mathbb{P}(\delta | \mathcal{D}) 
&\propto 
\left[var(x) + var(y) + \frac{1}{2}(\delta - (\bar{y} - \bar{x}))^2  \right]^{-(n - 1)} \\
&\propto
\left[ 1 + \frac{(\delta - (\bar{y} - \bar{x}))}{2(var(x) + var(y))} \right]^{-(n - 1)}
\end{align*}
$$

To then get the probability that $\delta > 0$, we can integrate over the pdf.

### Cauchy distribution

Imagine an archer standing infront of a target and shooting at random at a target, can we infer where the archer was originally standing depending on where the arrows landed?
The archer shoots in a random direction $ \mathbb{P}(\theta)d\theta = \frac{d\theta}{\pi}$ for $\theta \in  [-\frac{\pi}{2}, \frac{\pi}{2}]$. 
The middle is given by $\mu$ and the distance from the middle is given by $x - \mu$.
With $\theta = \arctan \left[ \frac{x - \mu}{L} \right]$ we then get the transformation 

$$
\mathbb{P}(x | \mu, L) dx = \mathbb{P}(f(x) | \mu, L) \left| \frac{df}{dx} \right| dx = \left| \frac{d\theta}{dx} \right| \frac{dx}{\pi} = \frac{1}{\pi L} \left( 1 + \left( \frac{x - \mu}{L}  \right)^2 \right)^{-1}
$$

This is the so called cauchy distribution $ \mathbb{P}(x | \mu = 0, L=1) = \frac{1}{pi(1 + x^2)}$. 
Looking at the cumulants of the cauchy distribution

$$
\langle x \rangle = \int_{\infty}^{\infty} \frac{x}{\pi(1 + x^2)}dx = \text{does not converge}
$$

The same follows for all the higher cumulants of the cauchy distribution. Thus it has no center of mass. It holds that the distribution of the average is the same as the distribution of each individual sample, that is the averaging holds the same information as each single sample about the center.
The likelihood of our observed data is

$$
\mathbb{P}(\mathcal{D} | \mu, L) = \prod_{i = 1}^{n} \mathbb{P}(x_i | \mu, L)
$$

Using a uniform prior we get our posterior

$$
\mathbb{P}(\mu, L | \mathcal{D}, I) = \frac{\prod_{i = 1}^{n} \left( \frac{1}{1 + \left( \frac{x_i - \mu}{L} \right)^2} \right)^{-1}}{ \int\int \prod_{i = 1}^{n} \left( \frac{1}{1 + \left( \frac{x_i - \mu}{L} \right)^2} \right)^{-1} d\mu dL}
$$

We can look at the distributions of $L$ and $\mu$ by integrating over one another. Imaging now setting L = 1, then we get the posterior

$$
\mathbb{P}(\mathcal{D} | \mu, L) \propto \prod_{i = 1}^{n} \frac{1}{1 + (x_i - \mu)^2 }
$$

To find the optimum value $\mu^*$, thus the maximum of the function, we can take it's derivative and set it to zero. Looking at the log posterior

$$
\frac{d \log [\mathbb{P}(\mu | \mathcal{D}, I)]}{d \mu} 
= 
\Rightarrow \sum_{i=1}^{n} \frac{d \log[1 + (x_i - \mu)^2]}{d\mu} 
= 
\sum_{i=1}^{n} \frac{(x_i - \mu)}{1 + (x_i - \mu)^2} \overset{!}{=} 0
$$

If we define $w_i = \frac{1}{1 + (x_i - \mu)^2}$, we get

$$
\sum_{i=1}^{n} (x_i - \mu)w_i \Leftrightarrow  \sum_{i=1}^{n} x_i w_i = \sum_{i=1}^{n} \mu w_i \Leftrightarrow \mu^* = \frac{\sum_{i=1}^{n} x_i w_i}{\sum_{i=1}^{n} w_i}
$$
This can be seen as the weighing the sample points and averaging them. This is impossible to analytically calculate as the weights themselves depend on $x$.

### Gene expression measurement noise

The difference between true log-exporession of the gene and the logarithm of the measured intensity is roughly gaussina distributed

$$
\mathbb{P}(\log(I) | \log(R),\mu, \sigma) = \frac{1}{\sigma \sqrt{w \pi}} \exp \left[ - \frac{1}{2} \left( \frac{\log(I) - \log(R)}{\sigma} \right)^2 \right]
$$

Because there is some extra noise in our measurements, we get outliers in our data. 
Thus, we get a mix of two distributions, one being the true distribution and the other one the noise distribution.
We create a so called outlier model to find out what the true mean of the true distribution is, where $\rho$ can be seen as the probability of favoring one distribution over the other. 

$$
\mathbb{P}(\mathcal{D} | \mu, \sigma, \mu_n, \sigma_n, \rho) = \prod_{i=1}^{n} \left[ \frac{\rho}{\sigma \sqrt{2\pi}} e^{- \frac{1}{2} \left( \frac{x_i - \mu}{\sigma} \right)^2} + \frac{1 - \rho}{\sigma_n \sqrt{2\pi}} e^{- \frac{1}{2} \left( \frac{x_i - \mu_n}{\sigma_n} \right)^2}\right]
$$

Assuming we know $\sigma, \sigma_n, \mu_n$, we want to find $\mu$ which maximizes the likelihood. Looking at the derivative of the log likelihood. We define $L_i(\mu) = \frac{\rho}{\sigma \sqrt{2\pi}} e^{- \frac{1}{2} \left( \frac{x_i - \mu}{\sigma} \right)^2}$ and $\tilde{L}_i = \frac{1 - \rho}{\sigma_n \sqrt{2\pi}} e^{- \frac{1}{2} \left( \frac{x_i - \mu_n}{\sigma_n} \right)^2}$ we get

$$
\frac{d \sum_{i=1}^{n} \log [L_i(\mu) \rho + \tilde{L}_i (1 - \rho)]}{d \mu} = \sum_{i=1}^{n} \frac{- \frac{x_i - \mu}{\sigma^2} L_i (\mu)}{ L_i(\mu) \rho + \tilde{L}_i (1 - \rho)} \\
\Rightarrow 
\sum_{i=1}^{n} \frac{\mu L_i (\mu)}{  L_i(\mu) \rho + \tilde{L}_i (1 - \rho)}
= 
\sum_{i=1}^{n} \frac{ x_i L_i (\mu)}{  L_i(\mu) \rho + \tilde{L}_i (1 - \rho)}
$$

If we now define $p_i = \frac{L_i(\mu) \rho}{L_i(\mu) \rho + \tilde{L}_i (1 - \rho)}$, which can be seen as the probability that $x_i$ belongs to the $L_i(\mu)$ distribution (good measurement). 
Because $ \mathbb{P}(good) = \rho$, we get

$$
\begin{align*}
\mathbb{P}(good | x_i, \mu) 
= 
\frac{\mathbb{P}(good, x_i, \mu)}{\mathbb{P}(x_i, \mu)} \\
&= 
\frac{\mathbb{P}(x_i | good, \mu) \mathbb{P}(\mu) \mathbb{P}(good)}{\mathbb{P}(x_i | good, \mu) \mathbb{P}(\mu) \mathbb{P}(good) + \mathbb{P}(x_i | bad, \mu) \mathbb{P}(\mu) \mathbb{P}(bad)} \\
&=
\frac{\mathbb{P}(x_i | good, \mu) \mathbb{P}(good)}{\mathbb{P}(x_i | good, \mu) \mathbb{P}(good) + \mathbb{P}(x_i | bad, \mu) \mathbb{P}(bad)} \\
&=
\frac{\mathbb{P}(x_i | good, \mu) \rho}{\mathbb{P}(x_i | good, \mu) \rho + \mathbb{P}(x_i | bad, \mu) (1 - \rho)} \\
&=
\frac{L_i(\mu) \rho}{L_i(\mu) \rho + \tilde{L}_i (1 - \rho)} \\
\end{align*}
$$


Thus:

$$
\sum_{i=1}^{n} \mu p_i
= 
\sum_{i=1}^{n} x_i p_i \\
\Rightarrow
\mu^* = \frac{\sum_{i=1}^{n} x_i p_i}{ \sum_{i=1}^{n} p_i}
$$

Which is just the measurement weighed by the probability that they belong the true distribution. Here the weights also depend on $\mu$ itself.

### Expectation Maximization
1. Start with initial value for $\mu$
2. Determine the likelihoods $L_i(\mu)$ and $\tilde{L}_i$
3. Calculate $p_i$
4. Determine new value $\mu$
5. If $\mu$ changed, go back to step 2 else we have found the optimal value $\mu^*$

This procedure is guaranteed to converge atleast to a local optimum of the likelihood.
