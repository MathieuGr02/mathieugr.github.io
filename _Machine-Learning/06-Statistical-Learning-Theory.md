---
title: Statistical Learning Theory
layout: collection
permalink: /Machine-Learning/Statistical-Learning-Theory
collection: Machine-Learning
entries_layout: grid
mathjax: true
toc: true
categories:
  - study
tags:
  - mathematics
  - statistics
  - machine-learning 
---

### Frequentist Decision theory

Let $\mathbf{x}$ be an i.i.d generated sample from an unknown pdf $\mathbb{P}^\*(\mathbf{x})$ and the output values $y$ be from the unknown conditional pdf $\mathbb{P}^\*(y | \mathbf{x})$.
The language model is then trained on a set of observed pairs drawn from the true unknown pdf given by $\mathbb{P}^\*(\mathbf{x}, y)$.
With this we get our data $\mathcal{D}=\{(\mathbf{x}_1, y_1),...\} \overset{i.i.d}{\sim} \mathbb{P}^\*(\mathbf{x}, y)$. 
We assume here that $\mathbf{x}$ is observable, but $y$ isn't. 
We then consider an estimator $\delta(\mathcal{D})$, which is defined as a prediction function $\hat{y} = f\_{\mathcal{D}}(\mathbf{x})$.
Since the data is random, so is the estimator, meaning the estimator is a RV. 
We can thus compare the estimates class membership with the true label via a loss function $L(y, f(\mathbf{x})$.
The expected risk is then the average loss with respect to the true unknown joint distribution $\mathbb{P}^*(\mathbf{x}, y)$, or it can be thought of as expectable error. 

$$
R(f, \mathbb{P}^*) 
= 
\mathbb{E}_{\mathbb{P}^*(\mathbf{x}, y)}L(y, f(\mathbf{x}) 
= 
\mathbb{E}_{\mathbb{P}^*(\mathbf{x})} \mathbb{E}_{\mathbb{P}^*(y | \mathbf{x})} L(y, f(\mathbf{x})
$$

Because the expected risk itself depends on the unknown distribution $\mathbb{P}^*$, and we don't know the true nature of $\mathbb{P}^*$ we cannot calculate this risk directly. We can approximate it with the so called emperical distribution for the n observed samples in $\mathcal{D}$:

$$
\mathbb{P}_{\mathcal{D}}(\mathbf{x}, y | \mathcal{D}) = \frac{1}{n} \sum_{(\mathbf{x}_i, y_i) \in \mathcal{D}} \delta({\mathbf{x} - \mathbf{x}_i) \delta(y,y_i)
$$

The emperical risk, which is the sample average of the loss is then given by 

$$
R_{emp}(f | \mathcal{D}) = \frac{1}{n} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i))
$$

### Hypothesis space

The best possible risk is defined as $\text{inf}_f R[f]$. Because we often have to restrict our function space, thus our hypothesis space $\mathcal{H}$, which is only a subset of the true space. 
In our given hypothesis space we define $f^*$ as the best possible function that can be implemented by a machine.

$$
f^* = \arg \min_{f \in  \mathcal{H}}
$$

We also denote $f_{\mathcal{D}} \in \mathcal{H}$ as the empirical risk minimizer on a sample $\mathcal{D}$

$$
f_n = f_{\mathcal{D}} = \arg \min_{f \in \mathcal{H}} R_{emp}(f | \mathcal{D})
$$

### Convergence

SLT gives then results for bounding the error on the unseen data, given the training data. Thus a relation between the past ($\mathcal{D}$) and the future (unseen data). In SLT all samples are i.i.d. We can bound the risk which with a probability of $1 - \delta$ holds with $a, b > 0$:

$$

R[f_n] \leq R_{emp} + \sqrt{\frac{a}{n}\left( \text{capacity}(\mathcal{H}) + \ln \frac{b}{\delta} \right)}

$$

Ont he covnerge of RV themselves, we say that X_n converges in probability to the random variable X as $n \rightarrow \infty$, iff, for all $\epsilon > 0$, 

$$
\mathbb{P}(|X_n - X| > \epsilon) \rightarrow 0, \text{as } n \rightarrow \infty
$$

We define this as $X_n \overset{p}{\rightarrow} X$ as $n \rightarrow \infty$.

### Binary classification

In general here we only look at the binary case where $f : X \rightarrow \{-1, 1\}$ with $L(y, f(\mathbf{x}) = \frac{1}{2}|1 - f(\mathbf{x})y|$. 
The hypothisis space is then $\mathcal{H}' = \{f' = \text{sign}(f) | f \in \mathcal{H}\}$

### Consistency of Emperical risk minimizations (ERM)

The principle of ERM is consistent if for any $\epsilon > 0$, 

$$
\lim_{n \rightarrow \infty} \mathbb{P}(|R[f_n] - R[f^*]| > \epsilon) = 0
$$

and 

$$
\lim_{n \rightarrow \infty} \mathbb{P}(|R_{emp}[f_n] - R[f_n]| > \epsilon) = 0
$$

Meaning that for $n \rightarrow \infty$ for the first case, the probability fo the true risk $R[f_n]$ deviating from the best possbile risk R[f^*] in our hypothisis space becomes zero, which means that the true risk of $f_n$ converges to the risk of the best possible function. The seconds case describes that the emperical risk, which is estimated from the data converges to it's true risk. 

![consistency erm](consistency_ERM.png)

Only the condition $\mathbb{P}(|R_{emp}[f] - R[f^*]|)$ does not suffice as a condition for consistency.

### Hoeffding's inequality

With Hoeffding's inequality we are able to bound our probability.
Let $\xi_i , i \in [0, n]$ be independent instances of a bounded RV $\xi$, with values in $[a, b]$. Denote their average by $ Q_n = \frac{1}{n} \sum_{i} \xi_i $. Then for any $ \epsilon > 0 $ we get:

$$
\begin{rcases}
\mathbb{P}(Q_n - \mathbb{E}(\xi) \geq \epsilon) \\
\mathbb{P}(\mathbb{E}(\xi) - Q_n \geq \epsilon)
\end{rcases}
\leq \exp \left( -\frac{2n\epsilon^2}{(b - a)^2} \right)
$$  

and 

$$
\mathbb{P}(|Q_n - \mathbb{E}(\xi)| \geq \epsilon) \leq 2 \exp \left( - \frac{2n\epsilon^2}{(b - a)^2} \right)
$$

Looking at our binary classification, we define $\xi$ to be a 0/1 loss function

$$
\xi = \frac{1}{2}|1 - f(\mathbf{x})y| = L(y, f(\mathbf{x}))
$$

Then we get that $ Q_n[f] = \frac{1}{n} \sum_{i=1}^{n} \xi_i = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(\mathbf{x}_i)) = R_{emp}[f] $ and $ \mathbb{E}[\xi] = \mathbb{E}[L(y, f(\mathbf{x}))] = R[f] $.
becuase $\xi_i$ are independent instances of a bounden RV $\xi$ with values in [0, 1] we get

$$
\mathbb{P}(|R_{emp}[f] - R[f]| > \epsilon) \leq 2 \exp \left( -2n\epsilon^2 \right)
$$

With this, the hoeffding's inequality gives us a rate of convergence for any fixed function. 
This doesn't tell us that $ \mathbb{P}(|R_{emp}[f_n] - R[f_n]| > \epsilon) \leq 2\exp \left(- 2n\epsilon^2 \right) $, because $f_n$ is not a fixed function. $f_n$ depends on the data $ \mathcal{D} $. 
Because $f_n$ is chosen to minimize the emperical risk, it may change with inceasing n so it is not a fixed function, therefor the hoeffding's inequality cannot be applied to this convergence.
For each fixed function $f$, we get $R_{emp}[f] \xrightarrow[n \rightarrow \infty]{P} R[f]$, meaning that the emperical risk converges to the true expected risk for a function as $n \rightarrow \infty$.

### Conditions for consistency

Let

$$
\begin{align*}
    f_n &:= \arg \min_{f \in \mathcal{H}} R_{emp}[f] \\
    f^* &:= \arg \min_{f \in \mathcal{H}} R[f] \\
\end{align*}
$$
then
$$
\begin{align*}
    R[f] - R[f^*] &\geq 0, \ \forall f \in \mathcal{H} \\
    R_{emp}[f] - R_{emp}[f_n] &\geq 0, \ \forall f \in \mathcal{H}
\end{align*}
$$

For first case, because $f^*$ is the function that minimizes the expected risk for all functions in the hypothisis space, any function in the hypothisis space has an equal or higher risk compared to $f^*$. 
For the second case, because $f_n$ is the function that minimizes the emperical risk for all functions in the hypothisis space, any function in the hypothisis space has an equal or higher emperical risk compared to $f_n$. 
Because this holds for any function $f \in \mathcal{H}$, we set $f = f_n$ for the first case and $f = f^*$ for the second one.
$$
\begin{align*}
    R[f_n] - R[f^*] &\geq 0 \\
    R_{emp}[f^*] - R_{emp}[f_n] &\geq 0
\end{align*}
$$

We can then write

$$
\begin{align*}
    0 
    &\leq 
    R[f_n] - R[f^*] + R_{emp}[f^*] - R_{emp}[f_n] \\
    &=
    R[f_n] - R_{emp}[f_n] + R_{emp}[f^*] - R[f^*] \\
    & \leq
    \sup_{f \in \mathcal{H}}(R[f] - R_{emp}[f]) + R_{emp}[f^*] - R[f^*] \\
\end{align*}
$$

If we assume that $\sup_{f \in \mathcal{H}}(R[f] - R_{emp}[f]) \xrightarrow[n \rightarrow \infty]{P} 0 $ (one sided uniform convergence over all functions in the hypothisis space) then:

$$
\sup_{f \in \mathcal{H}}(R[f] - R_{emp}[f]) + R_{emp}[f^*] - R[f^*] \xrightarrow[n \rightarrow \infty]{P} 0
$$

Which then means that by this assumption, this is a sufficient condition for consistency because the assumption implies the consistency of the ERM.

Let $\mathcal{H}$ be a set of functions wih bounded loss for the distribution $F(x, y)$, $A \leq R[f] \leq B, \ \forall f \in \mathcal{H}$. 
For the ERM principle to be consistent, it is necessary and sufficient that 
$$ 
\lim_{n \rightarrow \infty} \mathbb{P}(\sup_{f\in \mathcal{H}} (R[f] - R_{emp}[f]) > \epsilon) = 0, \ \forall \epsilon > 0
$$

