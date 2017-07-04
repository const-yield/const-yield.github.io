---
layout: post
title: "An Introduction to Structural Risk Minimization "
description: ""
category: Machine Learning 
tags: [Learning theory]
---

Supervised learning, or learning from examples,  refers to the training of a learning machine with examples.  The goal of supervised learning is to achieve good performance on test data. The performance of a learning machine evaluated on test data is termed to its *generalization* performance. 

The study of the generalization performance of a learning machine dates back to 1960s. 
It has been discovered that, for a given learning task with a finite set of training data, a learning machine can achieve its best generalization performance if a balance is struck between the accuracy attained on that particular training set and the complexity of the machine. This important criterion is formalized as the Structural Risk Minimization (SRM). 

This post we will give a brief introduction to structural risk minimization, a model selection framework for learning machines.  Before that we will brief about empirical risk and VC-dimension, the building blocks of the structural risk minimization theory.

Learning Machines 
------------------
Given a training set of $N$ instances { $(\mathbf{x_i},y_i)$ } $_{i=1}^{N}$, where $ \mathbf{x_i} \in X$ is an input and $y_i \in$ {-1, +1} is the corresponding response value (label),  the task of a learning machine is to learn the mapping: $\mathbf{x_i} \to y_i$.  A learning machine is defined by a family of functions $\{f( \mathbf{x},  \alpha)\}$ which are characterized by the adjustable parameters $\alpha$.   A particular choice of $\alpha$ determines a "trained machine".  Thus, for example, a linear classifier with fixed number of parameters, with $\alpha$ corresponding to the coefficients, is a learning machine.  

It is assumed that $(\mathbf{x},y)$ instances follow a joint probability distribution $P(\mathbf{x},y)$, and the instances $(\mathbf{x_i}, y_i)$ from the training set are drawn in an independent and identically distributed (i.i.d.) manner from $P(\mathbf{x},y)$. 

It is  also assumed that a real-valued loss function $L(y, \hat{y})$ is given which measures the discrepancy between the output $\hat{y}$ and the desired output $y$.

Empirical Risk 
------------------
The *expected risk* or the *risk* associated with a learning machine is defined as: 
\begin{equation}
R(\alpha) = E[L(f( \mathbf{x},  \alpha), y)] = \int L(f( \mathbf{x},  \alpha), y) dP(\mathbf{x},y)
\tag{1}
\end{equation}
From Eq. (1) we can see that *risk* measures how well  $f( \mathbf{x},  \alpha)$ fits the sample data drawn from the distribution $P(\mathbf{x},y)$ . It can be interpreted as a measure of test error for $f( \mathbf{x},  \alpha)$ . 

In general the joint probability distribution $P(\mathbf{x},y)$ is unknown and therefore we cannot calculate $R(\alpha)$.  However we can compute its approximation  *Empirical risk* , which is the average of loss values in the finite training set:
\begin{equation}
R_{emp}(\alpha) = \frac{1}{N} \sum_{i=1}^{N} L(f( \mathbf{x_i},  \alpha), y_i).
\tag{2}
\end{equation}
$R_{emp}(\alpha)$ can be considered as the measurement of training error for $f( \mathbf{x},  \alpha)$.  

Being an indirect measurement of $R(\alpha)$, $R_{emp}(\alpha)$ is not always consistent with $R(\alpha)$. When the learning machine $f( \mathbf{x},  \alpha)$ is simple, or in the case of under-fitting,  $R(\alpha)$ and $R_{emp}(\alpha)$ can be similar. But as $f( \mathbf{x},  \alpha)$ gets complex, over-fitting may occur. In this case the $R(\alpha)$ can be far worse than $R_{emp}(\alpha)$.  We therefore need a measure that describes the *complexity* of a learning machine.  VC-dimension is a quantitative way to measure the capacity of a learning machine, and it has been employed to analyze quite a few learning machines. 


VC Dimension
---------------------
VC dimension for Vapnik-Chervonenkis dimension measures the complexity of a learning machine that can be learned by statistical learning algorithms.  

To define VC dimension, we start with the term "shattering".  If a given training set of $l$ instances can be labelled in in all possible $2^{l}$ ways, and **at least one** member of the set $\{f( \mathbf{x},  \alpha)\}$ can achieve zero training error for each labeling, then we say that the set can be shattered by $\{f( \mathbf{x},  \alpha)\}$.  According to the definition, as long as there is at least one member in the set can achieve zero training error, i.e., $R_{emp}(\alpha) = 0$, then we can set the set is shattered by $\{f( \mathbf{x},  \alpha)\}$. 

Vapnik gave the defintion of VC dimension in his book [1]: "*the VC dimension of a set of indicator function $Q(z,\alpha)$ largest number h of vectors $z_1$, ..., $z_l$ that can be separated into two different class in all $2^{h}$  possible way using this set of functions (i.e., the VC dimension is the maximum number of vectors that can be shattered by the set of functions).*" In light of this definition, the VC dimension for a learning machine $\{f( \mathbf{x},  \alpha)\}$ is defined as the maximum size of a set shattered by it. 

Quite often we measure the VC dimension of a learning machine based on its definition. In other words, to show that a learning machine has a VC-dimension of $n$, we have to demonstrate it can shatter a set of size $n$ and it cannot shatter any set of size $n+1$. 
 
For example, to measure the VC dimension for the learning machine $\{f( \mathbf{x},  \alpha)\} = sign(x^{T}x-\alpha)$, we start with two training instances. Each instance is labeled with one of the two colors: red for positive and blue for negative. The following cases should be considered: (i) both instances have the same labels; (ii) the two instances have different labels. Both cases are drawn in the figure below. 

![A shattering example](/img/2017-03-30/shattering.png)
<center>Figure 1.  A shattering example </center>
 For case (i), if both instances are positive, $\alpha$ can be set to a small value such that both instances are  outside of the radius (See Fig 1.(a)) . On the other hand, if both instances are negative, $\alpha$ may be assigned a large value such that both instances are within the radius (See Fig 1.(b)). For case (ii), if the nearer of those two instances is negative and the further is positive, $\alpha$ can be adjusted so that the radius fall between them (See Fig 1.(c)). However if the nearer of them is positive and the other is negative (See Fig 1.(d)), there is no circle that we can create such that it labels the interior of the region red and the outer region blue.  That means no member from the set $\{f( \mathbf{x},  \alpha)\}$ can shatter two points.  Accordingly the VC dimension for $\{f( \mathbf{x},  \alpha)\}$ is just 1. 

In addition to measuring model complexity,  VC dimension can also be used to 
bound $R(\alpha)$ . Vapnik showed [2] that the following bound holds with probability $1-\eta$ where $\eta \in [0,1]$:

\begin{equation}
R(\alpha) \leq R_{emp}(\alpha) + \sqrt{\frac{h(\log(2l/h)+1)-log(\eta)/4}{l}}
\tag{3}
\end{equation}
where $h$ is the VC dimension of the learning machine and $l$ is the number of training samples.  

The inequality (3) defines a bound for the test error which is referred to as "VC bound". In inequality (3) the second term in the right hand side is regarded as "VC confidence". 

Note that :

 - The "VC confidence" is independent of $P(\mathbf{x}, y)$. 
 - It is usually not possible to compute the $R(\alpha)$ at the L.H.S
 - If $h$, i.e., the VC dimension, is known, then the R.H.S can be computed as an upper bound for the generalization performance of the learning machine 
 
This VC bound gives us a principled way of choosing learning machine for a given learning task. 


Structural Risk Minimization (SRM)
---

The idea of structural risk minimization is [1] to choose the machine "of the structure for which the smallest bound on the risk is achieved".

Structural risk minimization provides a quantitative way for model selection which selects models with a good balance between model complexity (VC dimension) and generalization performance. It does it by considering a bound of the risk, e.g., the VC bound in inequality (3). 

Note that the VC confidence in inequality (3) is indepdenent of $\alpha$ whereas both the risk and the empirical risk depend on $\alpha$.  To apply structural risk minimization one can firstly divide the family of functions into subsets.  The functions in all subsets have the same VC dimension $h$ and the same VC confidence as they are independent of $\alpha$. For each subset, one can then select the trained machine with minimum $R_{emp}(\alpha)$. By doing so  a series of trained machines can be obtained, one from each subset.  From that series, the trained machine with the minimum sum of $R_{emp}(\alpha)$ and VC confidence may be selected as the best trained machine for the learning task. 


----------

References 
---
[1] V. N. Vapnik  Statistical learning theory,  John Wiley & Sons, Inc.  
[2] V. Vapnik.The Nature of Statistical Learning Theory. Springer-Verlag, New York, 1995










