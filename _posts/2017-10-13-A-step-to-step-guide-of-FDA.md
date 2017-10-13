---
layout: post
title: "A steo-to-step guide of FDA"
description: ""
category: Dimensionality reduction
tags: [Machine Learning]
---

Introduction
------------
Dimensionality reduction is a fundamentally crucial for high-dimensional data anlysis. The goal of dimensionality reduction is to embed high-dimensional data into a low-dimensional embedding space, while retaining most of 'intrinsic information'. Amongst the numerous dimensionality reduction algorithms, Principal
Component Analysis (PCA) (Wold, 1987) and Fisher 
Discriminant Analysis (FDA) (Fisher, 1936) are the most classic approaches. Specifically, PCA works for unsupervised dimensionality reduction whilst FDA is designed for supervised dimensionality reduction. This post mainly focuses on FDA. 

Problem Formulation
-------------------
Given a data matrix $X$ = [$x_1$, $x_2$, $x_3$,..., $x_n$], where $x_i \in R^{d \times 1}$. The purpose of FDA is to learn a linear transformation matrix $W \in R^{d \times m} (m \ll d)$ which maps each $d$-dimensional $x_i$ to a $m$-dimensional $y_i$: $y_i = W^Tx_j $. 

LDA imposes a projection that pushes points from different classes away, while pulling points from the same class together. To specify such an objective, the following two matrices are defined:

__the between-class scatter matrix $S_b$__

$S_b = \sum_{i}^{C} n_i (\mu^{i} - \mu)(\mu^{i} - \mu)^T, \tag{1}$

where $n_i$ is the number of samples from the $i$-th class of $C$ classes, 
$\mu^i = \frac{1}{n_i} \sum_{i:y_i} x_i$ is the mean of the samples from the $i$-th class, and $\mu = \frac{1}{n} x_i$ is the sample mean.

__the within-class scatter matrix $S_w$__

$S_w = \sum_{i=1}^{C} \sum_{j=1}^{n_i} (x^{i}_j - \mu_i)(x^{i}_j - \mu_i)^T. \tag{2} $ 

The objective of FDA is defined as follows: 

$$\max_{W} \frac{ \sum_{i=1}^{C} n_i || W^T(\mu^i - \mu) ||^{2}_{2}} {\sum_{i=1}^{C} \sum_{j=1}^{n_i} || W^T(x^i_j - \mu_i)||^{2}_{2}} \tag{3}$$

     
The denominor from (3) can be written as:

$$\sum_{i=1}^{C} n_i  || W^T(\mu^i - \mu) || ^{2}_{2} $$

$$= \sum_{i=1}^{C} n_i (W^T(\mu^i - \mu))^T W^T(\mu^i - \mu) $$ 

$$= \sum_{i=1}^{C}  n_i (\mu^i - \mu)^T WW^T (\mu^i - \mu) $$ 

$$= \sum_{i=1}^{C} n_i W^T (\mu^i - \mu)(\mu^i - \mu)^T W $$  

$$= W^T ( \sum_{i=1}^{C} n_i (\mu^i - \mu)(\mu^i - \mu)^T ) W$$ 

$$= W^TS_bW$$

Similarly, the numerator from (3) can be transformed as:  

$$\sum_{i=1}^{C} \sum_{j=1}^{n_i} || W^T(x^i_j - \mu_i)||^{2}_{2}$$

$$= \sum_{i=1}^{C} \sum_{j=1}^{n_i} (W^T(x^i_j - \mu_i))^T( W^T(x^i_j - \mu_i)) $$ 

$$= \sum_{i=1}^{C} \sum_{j=1}^{n_i} (x^i_j - \mu_i)^TWW^T(x^i_j - \mu_i) $$

$$= \sum_{i=1}^{C} \sum_{j=1}^{n_i} W^T (x^i_j - \mu_i)(x^i_j - \mu_i)^TW$$

$$ = W^T \sum_{i=1}^{C} \sum_{j=1}^{n_i} (x^i_j - \mu_i)(x^i_j - \mu_i)^T W$$

$$ = W^TS_wW$$


As such (3) can be written as: 

$$\max_{W} \frac{W^TS_bW}{W^TS_wW}, \tag{4}$$
where both the denominator and numerator are $m \times m$ matrics. To convert them into scalar values, one possible way is to apply the trace operator: 

$\max_{W} \frac{ trace(W^TS_bW)}{trace(W^TS_wW)} \tag{5}$

We assume that the rank of $S_w$ is larger than $m$-$d$, i.e., the null spaces of $S_w$ has dimensionality less than $d$. This is to ensure the trace ratio value finite.  


Optimization 
-------------------

Note that if we scale $W$ by a scalar $k$, the ratio defined in (5) stays the same. As a result, we can fix the numerator whilst maxmizing the denomiator when searching for $W$.

Accordingly the optimization problem can be converted to the following:

$ \max_W  trace(W^TS_bW), s.t. trace(W^TS_wW) =1 \tag{6} $ 

To solve (6), we introduce a lagrange multiplier $\lambda$, which is essentially a $m \times m$  diagonal matrix since the trace operator returns the sum of diagonal elements):

$ L(W, \lambda) =  trace(W^TS_bW) - \lambda { trace(W^TS_wW) - 1} \tag{7}$

By taking the derivative of $L$ w.r.t $W$, we have: 

$$ \frac{\partial L}{\partial W} = S_BW+S_B^TW - \lambda(S_wW+S_w^TW) = 2S_bW - 2S_wW $$ 

since both $S_b$ and $S_w$ are symmetric matrices. 

By setting $\frac{\partial L}{\partial W}$ to zero, we can obtain:

$S_bW = \lambda S_wW \Longrightarrow S^{-1}_{w}S_b W = \lambda W \tag{8}$

(8) shows that $W$ is actually an eigenvector of $S^{-1}_{w}S_b$. 

The FDA algorithm
------------------- 

The FDA can be briefed into the following steps:   

 1. Compute the between-class scatter matrix $S_b$  
 
 2. Compute the within-class scatter matrix $S_w$   
 
 3. Select the top $k$ eigenvectors from $S^{-1}_{w}S_b$   
 
 4. Transform the data samples onto the embedding space 

We will walk through this process with the following Python codes. The Ipython notebook can be found [here](http://const-yield.github.io/notebooks/2017-10-13/Demonstration_of_Linear_Discriminant_Anysis_on_Sythetic_Data.ipynb).

#### Generate synthetic data points
Data points are sampled from three normal distributions.

```python
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
```
```python
matplotlib inline
```
```python
mu1, mu2, mu3 = [15,20], [24,25], [38,40]
cov = [[10, 0], [0, 10]]  
n_samples = 5000
```
```python
data1 = rand.multivariate_normal(mu1, cov, n_samples)
data2 = rand.multivariate_normal(mu2, cov, n_samples)
data3 = rand.multivariate_normal(mu3, cov, n_samples)
data = np.vstack((data1, data2, data3))
```
```python
plt.axis('equal')
plt.plot(data1[:,0], data1[:,1], '^b', label='Class_1')
plt.plot(data2[:,0], data2[:,1], 'sr', label='Class_2')
plt.plot(data3[:,0], data3[:,1], 'ok', label='Class_3')
plt.title('Original samples')
plt.legend(loc='best')
```
![png](/img/2017-10-13/original_data_clusters.png)


#### Compute mean values 

```python
mu1 = np.mean(data1, 0)
mu2 = np.mean(data2, 0)
mu3 = np.mean(data3, 0)
mu  = np.mean(data, 0)
```
 
#### Compute the between-class scatter matrix $S_b$ 

$S_b = \sum_{i}^{C} n_i (\mu^{i} - \mu)(\mu^{i} - \mu)^T$

```python
s1 = np.outer(mu1-mu, mu1-mu)*data1.shape[0] 
s2 = np.outer(mu2-mu, mu2-mu)*data2.shape[0] 
s3 = np.outer(mu3-mu, mu3-mu)*data3.shape[0] 
S_b = s1 + s2 + s3
```
   
#### Compute the within-class scatter matrix $S_w$ 

$S_w = \sum_{i=1}^{C} \sum_{j=1}^{n_i} (x^{i}_j - \mu_i)(x^{i}_j - \mu_i)^T$ 

```python
def compute_within_scatter_matrix(data, mu):
    """ 
    Compute the within-class scatter matrix for a given class
    :param data: a numpy matrix of (n_samples, n_sample_dimensions)
    :param mu:  a list of n_sample_dimensions 
    """
    matrix = np.zeros((data.shape[1], data.shape[1]))
    spread = data - mu
    for s in range(spread.shape[0]):
        matrix += np.outer(spread[s,:], spread[s,:])
    return matrix 
```
```python
s1 = compute_within_scatter_matrix(data1, mu1)
s2 = compute_within_scatter_matrix(data2, mu2)
s3 = compute_within_scatter_matrix(data3, mu3)
S_w = s1 + s2 + s3
```

#### Solve the generalized eigenvalue problem 

```python
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
```
```python
for eig_idx, eig_val in enumerate(eig_vals): 
    print('Eigvector #{}: {} (Eigvalue:{:.3f})'.format(eig_idx, eig_vecs[:, eig_idx], eig_val))
```

    Eigvector #0: [ 0.74758134  0.66417027] (Eigvalue:16.076)
    Eigvector #1: [-0.6710707   0.74139336] (Eigvalue:0.111)
    
#### Double-check the computed eigen-vectors and eigen-values 

```python
S = np.linalg.inv(S_w).dot(S_b)
for eig_idx, eig_val in enumerate(eig_vals):  
    eig_vec = eig_vecs[:, eig_idx]
    np.testing.assert_array_almost_equal(S.dot(eig_vec), eig_val*eig_vec, decimal=6, err_msg='', verbose=True)
```

#### Sort the eigenvectors by decreasing eigenvalues

```python
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted( eig_pairs, key=lambda x:x[0], reverse=True)
eigv_sum = sum(eig_vals)
for eig_val, eig_vec in eig_pairs:
    print('Eigvector: {} (Eigvalue:\t{:.3f},\t{:.2%} variance explained)'.format(eig_vec, eig_val, (eig_val/eigv_sum)))
```

    Eigvector: [ 0.74758134  0.66417027] (Eigvalue:	16.076,	99.32% variance explained)
    Eigvector: [-0.6710707   0.74139336] (Eigvalue:	0.111,	0.68% variance explained)
    

If we take a look at the eigenvalues, we can already see that the second eigenvalue are much smaller than the first one. 
Since Rank(AB) $\leq$ Rank(A), and Rank(AB) $\leq$ Rank(B), we have Rank($S_w^{-1}S_b$) $\leq$ Rank($S_b$). Due to that $S_b$ is the sum of $C$ matrices with rank 1 or less, Rank($S_b$) can be $C$-1 at most, where $C$ is the number of classes. This means that FDA can find at most $C$-1 meaningful features. The remining features discovered by FDA are arbitrary. 

#### Choose eigenvectors with the largest eigenvalues

After sorting the eigenpairs by decreasing eigenvalues, we can then construct our $d \times m$-dimensional transformation matrix $W$.
Here we choose the top most informative eigven-pair, as its eigenvalue explains 99.41% of the variance. As a result, the original d-dimensional (d=2) data points will be projected to a m-dimensional features space (m=1). 


```python
W = eig_pairs[0][1]
print('Matrix W:\n', W.real)
```

    ('Matrix W:\n', array([ 0.74758134,  0.66417027]))
    

#### Transform the samples onto the new space

As the last step, we use the 1 $\times$ 2 dimensional matrix $W$ to transform our samples onto the embedding space via the equation $Y = W^TX$. FDA learns a linear transformation matrix $W \in R^{d \times m} (m \ll d)$ which maps each $d$-dimensional (d=2)
$x_i$  to a $m$-dimensional (m=1) $y_i$: $y_i = W^Tx_j $. 

```python
X1_fda = W.dot(data1.T)
X2_fda = W.dot(data2.T)
X3_fda = W.dot(data3.T)
```

Now the transformed samples are scalar values. They are essentially the projection of the original data samples on the selected eigen vector, which corresponds to a straight line. To better visualize the projection, we visualize the transformed samples on the straight line under the original 2-dimensional space. 

```python
slope = W[1]/W[0] 
Y1_fda = slope * X1_fda
Y2_fda = slope * X2_fda
Y3_fda = slope * X3_fda 
```

```python
plt.axis('equal')
plt.plot(X1_fda, Y1_fda, '^b', label='Class_1')
plt.plot(X2_fda, Y2_fda, 'sr', label='Class_2')
plt.plot(X3_fda, Y3_fda, 'ok', label='Class_3')
plt.title('Projected samples')
plt.legend(loc='best')
```

![png](/img/2017-10-13/projected_data_clusters.png)

From the plot we can see that the projected samples retain most of the 'intrinsic information' from the original data samples. Dots with the same color stay together, while those with different colours stay away.



Discussions 
------------------- 
__FDA vs.PCA__ 

Both the Fisher Linear Discriminant Analysis (FDA) and Principal Component Analysis (PCA) are linear transformation techniques for dimensionality reduction. 

Principle Component Analysis (PCA) is a unsupervised dimensionality reduction method, which finds the axes with maximum variance (i.e., the "principal components") for the whole data set without any labelling information. In contrast, FDA finds the axes with maximum class separability (i.e., linear discrimanants). Although PCA is an unsupervised method, it can still be applied to supervised learnings. A common practise is to apply LDA followed by a PCA. 

__Single modality__

FDA might not work well for data with multi-modality. 


References
----------
(Sugiyama, 2007) Sugiyama, M. (2007). Dimensionality reduction of multimodal labeled data by local fisher discriminant analysis. Journal of machine learning research, 8(May), 1027-1061.

(Wold, 1987) Wold, S., Esbensen, K., & Geladi, P. (1987). Principal component analysis. Chemometrics and intelligent laboratory systems, 2(1-3), 37-52.

(Fisher, 1936) Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of human genetics, 7(2), 179-188.
