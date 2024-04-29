---
title: Sampling bayes theorem
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
from scipy.special import gammaln
from matplotlib import pyplot as plt
```

 Imagine we have a boxwith a total of $T$ balls in it, where among these, we have a total of $r$ red balls. If we now take out $n$ balls, how many of these are red?
We can imagine this as that each ball has a number from 1 to $T$ and we take out a ball one by one. Then for the first ball, we have a total of $T$ possible balls from which we can pick, the second time we only have $T - 1$ balls left etc. With this we get:

$$
T \cdot (T - 1) \cdot (T - 2) \cdot ... \cdot 2 \cdot 1 = T!
$$

Which is the total amount of arrangements of our numbered $T$ balls. Then taking only $n$ balls from $T$ gives us:

$$
T \cdot (T - 1) \cdot (T - 2) \cdot ... \cdot (T - n) = \frac{T!}{(T - n)!}
$$

From our first $n$ picked balls, these have $n!$ combinations we do not care about, so we divide by these amount of combinations to get out total amount of possibilities
of taking $n$ balls out of $T$:

$$
N = \frac{T!}{n!(T - n)!} = \begin{pmatrix} T \\ n \end{pmatrix}
$$ 

Because we then have no means of differentiating between these differnt possibilities, we get the uniform probability $\mathbb{P}(A_i \| I) = \frac{1}{N}$ for all these possibilities. If we now want to find out the probability of having $k$ red balls in our $n$ balls, we'd have to sum over all possibilities in $N$ where we have $k$ red balls:

$$
\mathbb{P}(k | I) = \frac{N(k)}{N}
$$

Looking at our $n$ picked balls, the amount of possibilities of having $n - k$ blue balls is given by $\begin{pmatrix} T - r \\ n - k \end{pmatrix}$ 
and the possibilities of having k red balls are $\begin{pmatrix} r \\ k \end{pmatrix}$
Thus in total we get

$$
N(k) = \begin{pmatrix} T - r \\ n - k \end{pmatrix} \begin{pmatrix} r \\ k \end{pmatrix}
$$

The probability of then having k red balls out of our sample of size $n$ from the total amount of balls $T$ is given by the so called hypergeomtric distribution: 

$$
\mathbb{P}(k | I) = \frac{\begin{pmatrix} T - r \\ n - k \end{pmatrix} \begin{pmatrix} r \\ k \end{pmatrix}}{\begin{pmatrix} T \\ n \end{pmatrix}}
$$

For large $T$ and smallr $n$ and $k$ we can approximate the factorial by a power multiplication, $(T - n)! \approx T^n$. By using this approximation we can transform our hypergeometric distribution into the so called Binomial distribution:

$$
\mathbb{P}(k | I) = \begin{pmatrix} n \\ k \end{pmatrix} \left( \frac{r}{T} \right)^k \left(1 - \frac{r}{T} \right)^{n - k}
$$


```python
p = 0.2
n = 20

def binomial(n, k, p):
  binomial_coefficient = np.exp(gammaln(n) - gammaln(k) - gammaln(n - k))
  bin = binomial_coefficient * (p**k) * (1 - p)**(n - k)
  return bin
k = range(21)
prob = [binomial(n, i, p) for i in k]
  
plt.figure(figsize=(5, 5))
plt.plot(k, prob)
plt.xlabel('k')
plt.ylabel('Probability')
plt.title('Binomial Distribution')
plt.show()
```


    
![png](/images/2024-04-27-sampling-bayes-theorem_files/2024-04-27-sampling-bayes-theorem_3_0.png)
    


### Mean, average, expected value

Given a distribution $\mathbb{P}(k \| I)$ the mean, average or expected value of this distribution is given as:

$$
\langle k \rangle = \sum_k k \mathbb{P}(K |n, p)
$$

For a fixed set, the mean is given as $\bar{k} = \frac{1}{n} \sum_{i = 1}^n k_i$

The mean of the binomial distribution is given as $\langle k \rangle = np$

<details>
<summary> Proof </summary>
\begin{align*}
\langle k \rangle 
&= \sum_k k \begin{pmatrix} n \\ k \end{pmatrix} p^k (1 - p)^{n - k} \\
&= \sum_k \begin{pmatrix} n \\ k \end{pmatrix} k \cdot p \cdot p^{k - 1} (1 - p)^{n - k} \\
&= \sum_k \begin{pmatrix} n \\ k \end{pmatrix} p \cdot \frac{d}{dq} q^k (1 - p)^{n - k} \Bigr|_{q = p} \\
&= p \frac{d}{dq} \sum_k \begin{pmatrix} n \\ k \end{pmatrix} q^k (1 - p)^{n - k} \Bigr|_{q = p} \\
&= p \frac{d}{dq} (1 - p + q)^n \Bigr|_{q = p} \\
&= np \quad q.e.d
\end{align*}
</details>

### Mode of the binomial distribution

The mode of a distribution is the most likely value, the value that most often appears in a given dataset of the distribution.

Let $k_*$ be the value of $k$ with the highest probability, the mode is given by 

$$
(n + 1)p - 1 < k_* < (n + 1)p
$$
