---
title: Quantitation Probabilities
layout: collection
permalink: /Computational-Biology/Quantitation-Probabilities
collection: Computational-Biology
entries_layout: grid
mathjax: true
toc: true
categories:
  - study
tags:
  - mathematics
  - statistics
---

For looking at probability of events, we set the following defintions. Let $A$ be an event and $I$ the 
given information:
- $\mathbb{P}(A \| I)$: The probability that $A$ is true given the information $I$.
- $\mathbb{P}(AB \| I)$: The probability that $A$ and $B$ are true given the information $I$.
- $\mathbb{P}(A \|B I)$: The probability that $A$ is true given the information $I$ and that $B$ is true. 

### Boolean Operations:

|  Not $A$ : $\bar{A}$  |  1  |  0  |
|:---------------------:|:---:|:---:|
|          $A$          |  0  |  1  |

| $A$ and $B$ : $AB$ | 0 | 0 | 0 | 1 |
|:------------------:|:-:|:-:|:-:|:-:|
|        $A$         | 0 | 0 | 1 | 1 |
|        $B$         | 0 | 1 | 0 | 1 |

| $A$ or $B$ : $A + B$ | 0 | 1 | 1 | 1 |
|:--------------------:|:-:|:-:|:-:|:-:|
|         $A$          | 0 | 0 | 1 | 1 |
|         $B$          | 0 | 1 | 0 | 1 |

| $A$ implies $B$ : $A \rightarrow B$ | 1 | 1 | 0 | 1 |
|:---------------------------------:|:-:|:-:|:-:|:-:|
|                $A$                | 0 | 0 | 1 | 1 |
|                $B$                | 0 | 1 | 0 | 1 |

The usage of such tables for larger systems is called the disjunctive normal form, where *AND's* and *OR's* are connected to make new statements.

# Rules of probabilities

1. Probabilites are real numbers between 0 (false) and 1 (true)
2. For false/true statements, the rules reduce to the rules of Boolean logic.
3. Consistency: If a probability can be derived in different was, it should give the same results.

The two main rules of probability, wherefrom every other rule can be derived are:
1. $$\mathbb{P}(A | I) + \mathbb{P}(\bar{A} | I)  = 1$$
2. $$ \mathbb{P}(AB | I) = \mathbb{P}(A | BI)\mathbb{P}(B | I) = \mathbb{P}(B | AI)\mathbb{P}(A | I)$$ 


For example we have

$$
\mathbb{P}(A + B | I) = \mathbb{P}(A | I) + \mathbb{P}(B | I) - \mathbb{P}(AB | I)
$$

<details>
<summary> Proof </summary>
$$
\begin{align*}
\mathbb{P}(A + B | I) 
&=\mathbb{P}(\overline{\bar{A}\bar{B}} | I) \\
&=1 - \mathbb{P}(\bar{A}\bar{B} | I) \\
&= 1 - (\mathbb{P}(\bar{A} | \bar{B}I)\mathbb{P}(\bar{B} | I)) \\
&= 1 - (1 - \mathbb{P}(A | \bar{B}I))\mathbb{P}(\bar{B} | I)) \\
&= 1 - \mathbb{P}(\bar{B} | I) + \mathbb{P}(A | \bar{B}I)\mathbb{P}(\bar{B} | I) \\
&= \mathbb{P}(B | I) + (\mathbb{P}(\bar{B} | AI) \mathbb{P}(A |I)) \\
&= \mathbb{P}(B | I) + (1 - \mathbb{P}(B | AI)) \mathbb{P}(A |I) \\
&= \mathbb{P}(B | I) + \mathbb{P}(A |I) - \mathbb{P}(B | AI)\mathbb{P}(A |I) \\
&= \mathbb{P}(B | I) + \mathbb{P}(A |I) - \mathbb{P}(AB | I) \quad q.e.d
\end{align*}
$$
</details>

Now lets imagine we have $n$ possible outcomes, $A_1, ..., A_n$, where each event is mutually exclusive,
 meaning that if $A_i$ is true, then all others are false and that they are exhaustive, 
 meaning that one out of all events must be true. With this we get the following properties:

$$
\mathbb{P}(A_i A_j | I) = 0 \ \forall i \neq j \quad \text{and} \quad \sum_{i = 1}^n \mathbb{P}(A_i | I) = 1
$$

Because we provide no further information with which we can distinguish event $A_i$ from $A_j$, we treat them all as the same and we finally get:

$$
\mathbb{P}(A_i | I) = \frac{1}{n} \ \forall i
$$
