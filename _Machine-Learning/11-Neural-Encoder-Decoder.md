---
title: Neural Encoder Decoder
layout: collection
permalink: /Machine-Learning/Neural-Encoder-Decoder
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


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
```

# Non-Linear Latent variable

In a non-linear latent variable model out likelihood is gaussian disributed with a non-linear transformed mean vector 

$$
\mathbf{\mu} = \mathbf{f} (\mathbf{z} , \phi ) 
$$

The prior and likelihood then look like

$$
\begin{align*}
    \mathbb{P}(\mathbf{z}) &= \mathcal{N}(0, I) \\
     \mathbb{P}(\mathbf{x} | \mathbf{z}, \phi  ) &= \mathcal{N}(f(\mathbf{z}, \phi ), \sigma^2 I ) 
\end{align*}
$$

Given an observation $ \mathbf{x}  $ we would then like to understand which hidenn latent variables we're responsible for the creation of $ \mathbf{x}  $, this is given by the posterior:

$$
\mathbb{P}(\mathbf{z} | \mathbf{x} ) = \frac{\mathbb{P}(\mathbf{x} | \mathbf{z} )\mathbb{P}(\mathbf{z})}{\mathbb{P}(\mathbf{x} )}  
$$

There exist no closed form form the posterior as the mean is a non-linear function. We can also not evaluate the evidence.

But sampling from this model is easy, we just draw a latent variable from the prior, pass it through our non-linear function to get the mean and then draw $\mathbb{x}$ with this mean from the likelihood.

Looking at the marginal likelihood of the evidence

$$
\begin{align*}
    \mathbb{P}(\mathbf{x} | \phi ) 
    &= 
    \int \mathbb{P}(\mathbf{x}, \mathbf{z} | \phi ) d \mathbf{z}  \\
    &=
    \int \mathbb{P}(\mathbf{x} | \mathbf{z}, \phi  ) \mathbb{P}(\mathbf{z} ) d \mathbf{z} \\
    &=
    \int \mathcal{N}(\mathbf{f}(\mathbf{z}, \phi ), \sigma^2 I ) \mathcal{N}(0, I) d \mathbf{z}     
\end{align*}

$$

Because $f$ is thus an arbitrary function, this integral doesn't have a closed form, but we can approximate it using the jensens inequality.

$$
\begin{align*}
    \log[\mathbb{P}(\mathbf{x} | \phi ) ] 
    &= 
    \log \left[\int \mathbb{P}(\mathbf{x}, \mathbf{z} | \phi ) d \mathbf{z}  \right] \\
    &=
    \log \left[\int q(\mathbf{z})  \frac{\mathbb{P}(\mathbf{x}, \mathbf{z} | \phi )}{q(\mathbf{z} )}  d \mathbf{z}  \right] \\
    &\geq
    \int q(\mathbf{z})  \log \left[ \frac{\mathbb{P}(\mathbf{x}, \mathbf{z} | \phi )}{q(\mathbf{z} )}\right]  d \mathbf{z}   \\
\end{align*}

$$

This holds true for any distribution $q$. This lower bound is called the evidence lower bound (ELBO). 
We assume that the distribution $q$ has some parameters $ \mathbf{\theta}  $. 
The ELBO then is given as 

$$
ELBO[\mathbf{\theta}, \phi ] = \int q(\mathbf{z}) \log \left[ \frac{\mathbb{P}(\mathbf{x}, \mathbf{z} | \phi  ) }{q(\mathbf{z} | \mathbf{\theta}  )}  \right] d \mathbf{z} 
$$

Becuase we want the tighest lower bound. i.e. approximate our evidence as best as possible, we would thus like to maximize the ELBO as a function of $ \mathbf{\theta}  $ and $\phi$.

$$
\begin{align*}
    ELBO[\mathbf{\theta}, \phi]
    &=
    \int q(\mathbf{z}, \mathbf{\theta}) \log \left[ \frac{\mathbb{P}(\mathbf{x}, \mathbf{z} | \phi)}{q(\mathbf{z}| \mathbf{\theta})} \right] d \mathbf{z} \\
    &=
    \int q(\mathbf{z}, \mathbf{\theta}) \log \left[ \frac{\mathbb{P}(\mathbf{z} | \mathbf{x}, \phi) \mathbb{P}(\mathbf{x} | \phi)}{q(\mathbf{z}| \mathbf{\theta})} \right] d \mathbf{z} \\
    &=
    \int q(\mathbf{z} | \mathbf{\theta}) \log [\mathbb{P}(\mathbf{x} | \phi)] d \mathbf{z} + \int q(\mathbf{z} | \mathbf{\theta}) \log \left[ \frac{\mathbb{P}(\mathbf{z} | \mathbf{x}, \phi)}{q(\mathbf{z}|\mathbf{\theta})} \right] d \mathbf{z} \\
    &=
    \log [\mathbb{P}(\mathbf{x} | \phi)] + \int q(\mathbf{z} | \mathbf{\theta}) \log \left[ \frac{\mathbb{P}(\mathbf{z} | \mathbf{x}, \phi)}{q(\mathbf{z}|\mathbf{\theta})} \right] d \mathbf{z} \\
    &=
    \log [\mathbb{P}(\mathbf{x} | \phi)] - \mathbb{KL}[q(\mathbf{z}|\mathbf{\theta}) || \mathbb{P}(\mathbf{z}|\mathbf{x}, \phi)]
\end{align*}
$$

This is maximized when we have $q(\mathbf{z}|\mathbf{\theta}) = \mathbb{P}(\mathbf{z}|\mathbf{x}, \phi) $. 
We can also write a different expression for the ELBO

$$
\begin{align*}
    ELBO[\mathbf{\theta}, \phi] 
    &= 
    \int q(\mathbf{z} |\mathbf{\theta} ) \log \left[ \frac{\mathbb{P}(\mathbf{x}, \mathbf{z} | \phi)}{q(\mathbf{z}|\mathbf{\theta})} \right] d \mathbf{z} \\
    &=
    \int q(\mathbf{z} |\mathbf{\theta} ) \log \left[ \frac{\mathbb{P}(\mathbf{x} | \mathbf{z}, \phi) \mathbb{P}(\mathbf{z})}{q(\mathbf{z}|\mathbf{\theta})} \right] d \mathbf{z} \\
    &=
    \int q(\mathbf{z} | \mathbf{\theta}) \log [\mathbb{P}(\mathbf{x} | \mathbf{z}, \phi)] d \mathbf{z} + \int q(\mathbf{z}|\mathbf{\theta}) \log \left[ \frac{\mathbb{P}(\mathbf{z})}{q(\mathbf{z}|\mathbf{\theta})} \right] d \mathbf{z} \\
    &=
    \int q(\mathbf{z} | \mathbf{\theta}) \log [\mathbb{P}(\mathbf{x} | \mathbf{z}, \phi)] d \mathbf{z} - \mathbb{KL}(q(\mathbf{z}|\mathbf{\theta}) || \mathbb{P}(\mathbf{z}))
    
\end{align*}
$$

The first term is an expectation in respect to q and we compute the expected value of the log-likelihood.
This term can be seen as how likely is the data given our model.
It is the average agreement of the data and the hidden variable.

The second term is a measure of how the prior and auxiliary distribution match.


Thus in summary, the ELBO is tight when we choose our auxiliary distribution to be our posterior

$$
q(\mathbf{z}|\mathbf{\theta}) = \mathbb{P}(\mathbf{z} | \mathbf{x}, \phi )
$$

But now because the posterior itself is intractable, we choose a simple parametric form for our auxiliary function and use that as an approximate to the true posterior.
Here we choose a normal distribution with mean $ \mathbf{\mu} $ and $\Sigma = \sigma^2 I$.

the optimization is then finding a normal distribution which is the closest to the true posterior, which corresponds to minimizing the KL divergence.
Because the ELBO is just 

$$
\underbrace{\log[\mathbb{P}(\mathbf{x}|\phi)]}_{\text{Evidence}} -
\underbrace{\mathbb{KL}[q(\mathbf{z}|\mathbf{\theta}) || \mathbb{P}(\mathbf{z}|\mathbf{x}, \phi)]  }_{\text{KL Divergence}}
$$

This term is then maximised by minimizing the KL divergence which means finds $q$ which is the closest to $\mathbb{P}(\mathbf{z}|\mathbf{x}, \phi)$, i.e. the posterior.

Now because the true posterior depends on our data $ \mathbf{x} $, the auxiliary function should logically then also depend on our data

$$
q(\mathbf{z}|\mathbf{\theta}, \mathbf{x}) = \mathcal{N}(g_{\mu}[\mathbf{x}|\mathbf{\theta}], g_{\sigma^2}[\mathbf{x}|\mathbf{\theta}])
$$

where $g[\mathbf{x}|\mathbf{\theta}]$ is a neural network with parameters $ \mathbf{\theta} $. Such a network is called a ***Variational Autoencoder***.

# Variational Autoencoder

Recall that 

$$
ELBO[\mathbf{\theta}, \phi] = \int q(\mathbf{z} | \mathbf{\theta}) \log [\mathbb{P}(\mathbf{x} | \mathbf{z}, \phi)] d \mathbf{z} - \mathbb{KL}(q(\mathbf{z}|\mathbf{\theta}) || \mathbb{P}(\mathbf{z}))
$$

The first term still holds an intractable integral but because this is an expectation we can approximate it using sampling

$$
\mathbb{E}_{q(\mathbf{z}|\mathbf{x}, \mathbf{\theta})} [\log [\mathbb{P}(\mathbf{x}|\mathbf{z}, \phi)]] 
\approx
\frac{1}{N} \sum_{i=1}^N \log[\mathbb{P}(\mathbf{x}|\mathbf{z}^*_n, \phi)]
$$

where $ \mathbf{z}_n^*$ is the n-th sample from $q(\mathbf{z}|\mathbf{x}, \mathbf{\theta})$. Often only one sample is used to approximate the expectation, which gives us

$$
ELBO[\mathbf{\theta}, \phi] 
\approx
\log[\mathbb{P}(\mathbf{x}|\mathbf{z}^*, \phi)] - \mathbb{KL}(q(\mathbf{z}|\mathbf{x}, \theta) || \mathbb{P}(z))
$$

# Reparameterization trick


Because we want to sample from the distribution

$$
q(\mathbf{z}|\mathbf{\theta}, \mathbf{x}) = \mathcal{N}(g_{\mu}[\mathbf{x}|\mathbf{\theta}], g_{\sigma^2}[\mathbf{x}|\mathbf{\theta}])
$$

But we would like to avoid this sampling step because stochastic sampling prevents the use of backproporgation to train our NN. The go around this, we sample noise from a fixed distribution $ \mathbf{\xi} \sim \mathcal{N}(0, I)$ and use the trick

$$
\mathbf{z}^* = g_{\mu} + \sigma^{1/2} \mathbf{\xi} 
$$

Which allows the to define the encoding network.

We also want to minimize the negative expectation of the ELBO over $ \mathbb{P}(\mathbf{x}) $

$$
\min_{\phi, \mathbf{\theta}} - \mathbb{E}_{\mathbb{P}(\mathbf{x})} \mathbb{E}_{q(\mathbf{z}|\mathbf{x}, \mathbf{\theta})} [\log[\mathbb{P}(\mathbf{x}|\mathbf{z}, \phi)]] + \mathbb{E}_{\mathbb{P}(\mathbf{x})} \mathbb{KL}(q(\mathbf{z}|\mathbf{x}, \theta) || \mathbb{P}(z))
$$

The first term is approximated as 

$$
\mathbb{E}_{\mathbb{P}(\mathbf{x})} \mathbb{E}_{q(\mathbf{z}|\mathbf{x}, \mathbf{\theta})} [\log[\mathbb{P}(\mathbf{x}|\mathbf{z}, \phi)]] 
\approx
\frac{1}{n} \sum_{i=1}^n \log [\mathbb{P}(\mathbf{x}_i | \mathbf{z}_i^*, \phi)]
$$

We assume $ \mathbb{P}(\mathbf{x}_i | \mathbf{z}_i^*, \phi) = \mathcal{N}(f_{\phi}(\mathbf{z}_i^*), \sigma^2) $
Where $f$ is then implemented via a NN.
In a escence a encoder, decoder structure is similair to PCA, but where we use non-linear transformations instead of our linear transformations $W$.

![vae](../images/vae_nn.png)

For maximizing the ELBO, we jointly optimize over the parameters of the encoder and decoder network.
But when changing the decoder we also change the true posterior we want to approximate, which creates a moving target.
When trying to minimize the negative expected ELBO, we add a 'tuning knob' which steers the model in the desired direction.
We could for example introduce $ \beta > 0 $ which controls the realative importence of the two loss terms

$$
\min_{\mathbf{\theta}, \phi} \frac{1}{n} \sum_{i=1}^n \mathbb{KL}(q(\mathbf{z}|\mathbf{x}_i, \mathbf{\theta}), \mathbb{P}(\mathbf{z})) - \beta \frac{1}{n} \sum_{i=1}^n \log[\mathbb{P}(\mathbf{x}_i | \mathbf{z}_i^*, \phi)]
$$

Which works as an emphasis factor on reconstruction compared to allignement.
For $\beta \rightarrow 0$ we get only the allignement term, which means that our auxiliary function, i.e. our posterior approximation would just resemble the prior, meaning that we haven't learned anything.
For $\beta \rightarrow \infty$ we get only our reconstruction error term, which would cause us to choose our variational approximation which would shrink to the posterior maximum.

# The deep information bottleneck

We want to find dependency structure between different views through nonlinear models. These dependencies are found by deep IB.
So far because our true posterior depended on $ \mathbf{x} $, we said that our variational approximation should aswell.
Image we're given another variable $ \mathbf{\tilde{x}} $ and our variational approximation only depends that external variable.

$$
q = q(\mathbf{z}|\mathbf{\theta}, \mathbf{\tilde{x}})
$$

$$
\begin{align*}
    ELBO[\mathbf{\theta}, \phi]
    &=
    \int q(\mathbf{z}|\mathbf{\tilde{x}}, \mathbf{\theta}) \log[\mathbb{P}(\mathbf{x}|\mathbf{z}, \phi)] d \mathbf{z} - \mathbb{KL}[q(\mathbf{z}|\mathbf{\tilde{x}}, \mathbf{\theta})||\mathbb{P}(\mathbf{z})]
\end{align*}
$$

Connection to the IB is that if we set $ q(\mathbf{z}|\mathbf{\tilde{x}}, \mathbf{\theta}) := \mathbb{P}(\mathbf{z}|\mathbf{\tilde{x}}, \mathbf{\theta}) $ and take the expectation with respect to the joint data distribution

$$
\mathbb{E}_{\mathbb{P}(\mathbf{\tilde{x}}, \mathbf{x})} \mathbb{E}_{\mathbb{P}(\mathbf{z}|\mathbf{\tilde{x}}, \mathbf{\theta})} \log[q(\mathbf{x}|\mathbf{z}, \phi)] - \mathbb{E}_{\mathbb{P}(\mathbf{x})} \mathbb{KL}[\mathbb{P}(\mathbf{z}|\mathbf{\tilde{x}}, \mathbf{\theta})||\mathbb{P}(\mathbf{z})]
$$

The first term is a lower bound on the mutual information $\leq \mathcal{I}_{\mathbf{\theta}, \phi}(\mathbf{z};\mathbf{x}) + const$ and the second term is the mutual information $= \mathcal{I}_{\mathbf{\theta}}(\mathbf{\tilde{x}}; \mathbf{z}) $.
Thus we minimize the mutual information between $ \mathbf{\tilde{x}} $ and $ \mathbf{z} $, meaning we want to compress $\mathbf{\tilde{x}}$ ,i.e. have a high compression rate ($\mathbf{z}$ should be a very compact representation), while also preserving the information between $ \mathbf{x} $ and $ \mathbf{z} $, i.e. have as much information in $\mathbf{z}$ about $ \mathbf{x} $ as possible.
This defines the deep information bottleneck

$$
\min_{\phi, \mathbf{\theta}} \mathcal{I}_{\mathbf{\theta}}(\mathbf{\tilde{x}}; \mathbf{z}) -\beta \mathcal{I}_{\mathbf{\theta}, \phi}^{\text{low}}(\mathbf{z}; \mathbf{x})
$$

# Seq2seq

A sequence to sequence modes is an example of a conditional language model

* Language model because the decoder is predicting the next word of the target sequence $ \mathbf{y} $
* Conditional because its predictions are also conditioned on the source sentence $ \mathbf{x} $

In neural machine translation we directly calculate

$$
\mathbb{P}(\mathbf{y}|\mathbf{x}) = \mathbb{P}(y_1 | \mathbf{x}) \cdot \mathbb{P}(y_2 | y_1, \mathbf{x}) \cdots \mathbb{P}(y_T | y_1, ..., y_{T-1}, \mathbf{x})
$$

Seq2seq is optimized as a single system where the model is trained on a big amount of text where backproporgation operates end-to-end. The loss is often calculated as a sum of the individual losses divided by the amount of targets, i.e. $L = \frac{1}{T}\sum_{i} L_i$.

![seq2seq](../images/seq2seq.png)

Often a problem with such seq2seq is because of their implementation via a RNN, the last input sequence holds a compressed summary of the whole input sequence. 
This causes an information bottleneck.
To bypass this bottleneck we could allow the decoder to directly look at the input sequence through so called attention

# Attention

In the encoder we have our different hidden states $ \mathbf{h}_1, ..., \mathbf{h}_N \in \mathbb{R}^h $, which we call our values and on timestep $t$, we have our decoder hidden state $ \mathbf{s}_t \in \mathbb{R}^h$, which we call our queries.
At each step $t$ we calculate our attention score, which is the dot product of our current query and all the values. 

$$
\mathbf{a}^t = (\mathbf{s}_t^T \mathbf{h}_1, ..., \mathbf{s}_t^T \mathbf{h}_N) \in \mathbb{R}^N
$$

This dot product measures in a sense the allignement or co-linearity between the query and values. 
We then take the softmax of every allignement score to get an 'attention distribution'. 

$$ 
\mathbf{\alpha}^t = \text{softmax}(\mathbf{a}^t)
$$

We user our attention distribution to take a weighted sum of the values to get the attention output

$$
\mathbf{z}_i = \sum_{i=1}^N \mathbf{\alpha}^t_i \mathbf{h}_i \in \mathbb{R}^h
$$

Then we concatenate the attention output with the query state $ [\mathbf{z}_t : \mathbf{s}_t] $.
We use this then to generate the probability of the next word $ \mathbb{P}(y_z | y_1, ..., y_{t-1}, \mathbf{x}) $ and predict the next word as $\hat{y}_t = \arg \max \mathbb{P}(y_z | y_1, ..., y_{t-1}, \mathbf{x})$.

![attention_rnn](../images/attention_rnn.png)

This attention solves the bottleneck problem as it allows the decoder (querie) to directly look at the input sequence instead of the final compressed hidden state, thus it has much more access to information to make its next prediction.
Attention also helps the vanishing gradient problem as it functions as a shortcut to faraway states.
What is also of advantage is that the architecture itself learns the allignements of words.

# Attention and KRR

We can try to understand the general principle of attention better by comparing it to kernel ridge regression (KRR) (combines regulare ridge regression with the kernel trick).
In KRR, we compare the input ('query) $ \mathbf{x} \in \mathbb{R}^d $ to each of the training examples $X = (\mathbf{x}_1, ..., \mathbf{x}_n)$ using a kernel function $ K(\mathbf{x}, \mathbf{x}') $to get a vector of similarity scores $ \mathbf{\alpha} = \mathbf{\alpha}(\mathbf{x}, X) $.
We then use these similairity scores to retrieve a weighted combination of the target values $ \mathbf{y}_i \in \mathbb{R}^{d_v} $ to compute the predicted output $ \mathbf{z} = \sum_{i=1}^n \alpha_i \mathbf{y}_i $.
For example for the one dimensional output targets $d_v = 1$. 
given the kernel function $K(\mathbf{x}, \mathbf{x}')$, the predicted regression output for query $ \mathbf{x}$ is 

$$
z := f(\mathbf{x}) = \underbrace{K^t_{\mathbf{x}}(K(X, X) + \lambda I)^{-1} }_{\mathbf{\alpha}^t} \mathbf{y} = \sum_{i=1}^n \alpha_i(\mathbf{x}, X) y_i
$$

with  $ K^t_{\mathbf{x}} = \left[K(\mathbf{x}, \mathbf{x}_1), K(\mathbf{x}, \mathbf{x}_2), ... \right] $. Thus  $ \mathbf{\alpha}(\mathbf{x}, X)  $ measures how well the query $ \mathbf{x} $ is aligned with the examples in the training set $X$.

![krr_attention](../images/krr_attention.png)

We could then replace our data $X$ with a learned embedding, to create a set of keys $K = X W_k \in \mathbb{R}^{n \times d_k}$ and replace $Y$ with a learned embedding, to create a set of values $V = Y W_v \in \mathbb{R}^{n \times d_v}$.
We also embed our input to create a query $q = W_q \mathbf{x} \in \mathbb{R}^{d_k}$.
The learnable parameters in this model are then these embedding matrices.
We could then replace the similairity scoring of the kernel ridge regression ($\mathbf{\alpha}(\mathbf{x}, X)$) with a soft attention layer.
We define the weighted output for our query $ \mathbf{q} $ to be

$$
\mathbf{z} := \text{Attn}(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), ..., (\mathbf{k}_n, \mathbf{v}_n)) = \sum_{i=1}^n \alpha_i(\mathbf{q}, K) \mathbf{v}_i
$$

where $\alpha_i(\mathbf{q}, K)$ is the i'th attention weight which satisfies $0 \leq \alpha_i \leq 1$ and $\sum_{i} \alpha_i = 1$. 
These attention weights work as weighting the different embeddings of our targets $ \mathbf{v}_i $ to create a new output.
The attention weights $ \alpha_i(\mathbf{q}, K) $ are computed from an attention score $a(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$, which computes the similarity between our query (embedded input) and the our keys (embedded data).
For example the attention score can be the previous described dot product $a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^T \mathbf{k} / \sqrt{d_k}$ with a normalization factor $\sqrt{d_k}$.
Given then the scores we can calculate the attention weights

$$
\alpha_i(\mathbf{q}, \mathbf{k}_{1:n}) = \text{softmax}_i ([a(\mathbf{q}, \mathbf{k}_1), ..., a(\mathbf{q}, \mathbf{k}_n) ]) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^n \exp(a(\mathbf{q}, \mathbf{k}_j))}
$$


![attention_weights](../images/attention_weights.png)

# Self Attention

Self attention is where we use the output of one attention layer as input to another. Here the keys, queries and values all come from the input $X$.

In self attention thus when using our input $X$ as keys, values and query, we can compare the input with itself and find semantic relationships inside our input.

![self_attention](../images/self_attention.png)

# Masked Attention

Makes attention is where we prevent vectors (queries) looking at future vectors (keys) by setting the similarity score to $-\infty$.

![masked_attention](../images/masked_attention.png)

# Multi-head attention

Multi head attention is just the several parallelisation of the attention mechanism. The independent attention outputs are then concatenated and linearly transformed into the expected dimension.
The use of several parallel attention mechanisms allows for attending to parts of the sequences differently (long term and versus short term). These attention layers are able to then learn semantic connections inside the input (sentence).

![multi_head_attention](../images/multi_head_attention.png)

# Transformers

In RNN's we process tokens sequentially that is, one after another. Thus the representation of a word at location $t$ depends on the hidden state $ \mathbf{s}_t $ which is a summary of all the previous words.
In contrast, we could just use attentiopn to compute representation directly as a function of all other words. 
We calculate our attention output $ \mathbf{z}_j $ of the j'th word as a function of all the other words in the sentence.
This idea is called an encoder-only transformer.
There exist also decoder-only transformers where each output $y_t$ only attends to all previously generated outputs $y_{1:(t-1)}$.

While the encoder only architectures are designed to learn embeddings that can be used for various predictive modeling tasks such as classification.
Decoders are designed to generate new text.

When combining encoder and decoder networks, we get the so called sequence to sequence models, $\mathbb{P}(y_{1:T_y} | x_{1:T_x})$

# Encoder

![encoder_transformer](../images/encoder_transformer.png)

MHSA often produces features at different scales or magnitudes.
Because the attention weights can be very different, combining multiple attention heads makes this even more problematic.
Because of this we add a normalization layer and a word wise feed forward MLP that updates the representation of the i'th word

$$
h_i \leftarrow \text{Norm}(\text{FeedFwd}(\text{Norm}(h_i)))
$$

rescaling the feature vectors independently of each other helps to overcome remaining scaling issues.

We also add residual connections which allow positional information to propagate to higher layers.

# Positional encoding

The processing of the feature vectors for computing query, key and value occurs in parallel. The order of information between the words is not known anywhere inside the attention block, thus we need positional encoding the keep the order of information as semantical information depends on this.
For this we add a vector to each input of a specific pattern which the model learns, that helps it to determine the position of each word or the distance between different words in the sequence.
For example we could use the positional encoding

$$
\mathbf{p}_t = 
\begin{bmatrix} 
    \sin(\omega_1 t) \\
    \cos(\omega_1 t) \\
    \sin(\omega_2 t) \\
    \cos(\omega_2 t) \\
    \vdots \\
    \sin(\omega_{d/2}) t) \\
    \cos(\omega_{d/2} t) \\

\end{bmatrix}
$$

# Seq2Seq with Transformers

Coupling of an encoder and decoder allows us in the encoder step to capture the semantic relations of the words inside the sentence and the decoder than uses these as inputs to create an output. 
May it be a translation from one language to another. 

![seq2seq_transformer](../images/seq2seq_transformer.png)

# Resources

https://jalammar.github.io/illustrated-transformer/
