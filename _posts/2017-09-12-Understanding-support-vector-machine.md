---
layout: post
title: "Undersetanding Support Vector Machine"
description: ""
category: Machine Learning 
tags: [SVM]
---

Support Vector Machine (SVM), is a widely used discriminative model for classification. This blog entry documented the insights developed in when learning SVM.


Mathematical Formulation 
-------------
The support vector machine (SVM) is a classification algorithm.  Initially it was used for binary classification but later on it was extended for multi-class classification. 


Let {$x_i$}, $i$=1,...,N be a set of $N$ input vectors, and $\\{y_i\\}$ be the corresponding target values, where $y_i \in \\{-1,1\\}$. Each pair of $(x_i, y_i)$ is considered as a training sample. The SVM classifies each input vector by learning the following linear model: 
\begin{equation}
   f(x) = \mathbf{w}^{T}\phi(x) + b 
  \tag{1}
\end{equation}
from the given training set which comprises $\{(x_i, y_i)\}$ pairs, where $\phi(.)$ stands for a fixed feature space transformation. A new input vector $x$ is classified according to the sign of $f(x)$.  

If we assume the training set is linear separable,  there then exists at least one choice of $(\mathbf{w}, b)$ such that $f(x_i) >0 $ for samples having $y_i = -1 $  and $f(x_i) <0$ for samples having $y_i = 1$. Each choice of $(\mathbf{w}, b)$ defines a hyper plane. As indicated in Figure 1, there may exist a number of such solutions that separate the training samples well ( both the dotted lines can separate the samples well). However we favor the one that has the smallest generalization error. 

<div>
<center>
<img  src="/img/2017-09-12/hyperplane.png">
</center>
<center>Figure 1. The hyperplanes </center>
<br>
</div>


The SVM approach tackles this problem by introducing the concept of "margin", which is defined as the smallest distance from any training sample to the hyper plane: 
\begin{equation}
\min_i \frac{y_i( \mathbf{w}^{T}\phi(x_i)+b)}{|| \mathbf{w}||} 
\tag{2}
\end{equation} 
The hyper plane that has the smallest generalization error has the largest margin. In figure 1, the solid line stands for such a hyper plane. 

As a result, we look for $W$ and $b$ which maximize (2): 
\begin{equation}
\max_{\mathbf{w},b}{ \{\min_i \frac{y_i( \mathbf{w}^{T}\phi(x_i)+b)}{|| \mathbf{w}||} \}}
\tag{3}
\end{equation}
Once $ \mathbf{w}$ and $b$ are optimised, we can locate the desired hyper plane. The goal of the optimisation is to convert (3) into an equivalent problem that is easier to solve. Firstly we can take $\frac{1}{|| \mathbf{w}||}$ outside the min operator in (3) because $\mathbf{w}$ does not rely on $i$. As such (3) can be written as: 
\begin{equation}
\max_{\mathbf{w},b} \{ \frac{1}{|| \mathbf{w}||} \min_i  y_i( \mathbf{w}^{T}\phi(x_i)+b)\}
\tag{4}
\end{equation}
Note that if we scale the parameters $(\mathbf{w},b)$ to $(k\mathbf{w}, kb)$ where $k$ is a scalar value, the margin defined by (2) will not change.  Based on this observation we can set $y_i( \mathbf{w}^{T}\phi(x_i)+b)$ to $1$ for the sample that is closest to the hyper plane. This can be done by scaling $(\mathbf{w},b)$ by an appropriate value.   By doing so,  we have $y_i( \mathbf{w}^{T}\phi(x_i)+b) \geq 1$ for $i =1,2,...N.$  

In light of the above mentioned reasoning, (4) can be simplified to  
\begin{equation}
\max_{\mathbf{w},b} \frac{1}{|| \mathbf{w}||}  \\
s.t.\forall i,  y_i( \mathbf{w}^{T}\phi(x_i)+b) \geq 1
\tag{5}
\end{equation}
Since maximising $\frac{1}{||\mathbf{w}||}$ is equivalent  to minimising $\frac{1}{2}||\mathbf{w}||^{2}$, the final objective function can then be formulated as:
\begin{equation}
\min_{\mathbf{w},b} \frac{1}{2}{|| \mathbf{w}||^2}  \\
s.t.\forall i,  y_i( \mathbf{w}^{T}\phi(x_i)+b) \geq 1
\tag{6}
\end{equation}

Lagrangian  Optimisation
-------------

The objective in (6) satisfies the following form: 
\begin{equation}
\min_{x} f(x) \\
s.t. g_i(x) \leq 0, i=1,...k; \\
h_j(x) = 0, j=1,...,l 
\tag{7}
\end{equation}

To solve the minimization problem in (7), we define the following terms: 

 - The Lagrangian $L(x, \alpha, \beta) := f(x) + \sum_i \alpha_i g_i(x) + \sum_j \beta_jh_j(x)$, where $\mathbf{\alpha, \beta}$ are termed the Lagrangian multipliers, and $\alpha_i \geq 0 $ 
 - $\theta_p(x) = \max_{\mathbf{\alpha, \beta}: \alpha_i \geq 0} L(x, \alpha, \beta) $. Obviously $\theta_p(x) = f(x)$  if $w$ satisfies all the constraints. 
 - $\theta_{D}( \mathbf{\alpha, \beta}) = \min_{x} L(x, \alpha, \beta)$

Using these terms, we define the $\bf{primal}$ and $\bf{dual}$ optimization problems for (7): 

 - The primal optimization problem is formulated as 
 $\min_{x} \theta_p(x) = \min_{x} \max_{\mathbf{\alpha, \beta}: \alpha_i \geq 0} L(x, \alpha, \beta)$. 

 - The dual optimization problem is formulated as $\max_{\mathbf{\alpha, \beta}: \alpha_i \geq 0} ?\theta_{D}( \mathbf{\alpha, \beta})  = \max_{\mathbf{\alpha, \beta}: \alpha_i \geq 0} ?\min L(x, \alpha, \beta).$ 

Strong Duality and Slater's Condition
-------------

Denote $ p^{\*}$  = $\min_{x} \theta_p(x) $ be the optimal $\bf{value}$ for the primal optimization problem, and $d^{\*}= \max_{\mathbf{\alpha, \beta}: \alpha_i \geq 0} \theta_{D}( \mathbf{\alpha, \beta})$ as the optimal $\bf{value}$ for the dual optimization problem.  

It can be easily seen that $d^{\*} \leq p^{\*}$,  and this inequality is termed the $\bf{weak}$ $\bf{duality}$. However under some certain conditions, we have $d^{\*}= p^{\*}$.  More specifically, if the [Slater's condition](https://en.wikipedia.org/wiki/Slater%27s_condition) is satisfied, then we have $d^{\*}= p^{\*}$, which is termed as the $\bf{strong}$ $\bf{duality}$.

The Slater's condition can be stated as: if the primal is a convex problem (i.e., $f$ and $g_1, g_2, ..., g_k$ are convex, $h_1, h_2, ..., h_l$ are affine), and there exist at least one $x$, which satisfies strictly negative inequality, meaning $g_i(x) \lt 0, i=1,...k$, and $h_j(x) = 0, j=1,...,l$., then strong duality holds. 
 
Note that the Slater's condition is a weak condition. And it is not hard to see that the problem in (6) satisfies the Slater's condition. And therefore, strong duality holds for (6). 

The Karush-Kuhn-Tucker (KKT) Condition 
-------------
For a general optimization of the form presented in (7),  the KKT conditions are: 

 - Stationarity: $0 \in \partial f(x) + \sum_{i=1}^{k} \alpha_i \partial g_i(x) +  \sum_{j=1}^{l} \beta_j \partial h_j(x) $
 - Complementary: $\alpha_ig_i(x) = 0, i=1,...k$
 - Primal feasibility: $ g_i(x) \leq 0, i=1,...k; h_j(x) = 0, j=1,...,l $
 - Dual feasibility: $\alpha_i \geq 0,  i=1,...k$ 

The KKT conditions are strongly associated with strong duality: for a problem with strong duality, $x^{\*}$, and $\alpha^{\*}$, $\beta^{\*}$ satisfy the KKT conditions iff $x^{\*}$, and $\alpha^{\*}$ $\beta^{\*}$ are the primal and dual solutions respectively. The proof of this theorem can be found in [1]. 

Parameter Learning 
-------------
Since duality holds for (6), we can use KKT conditions to find the primal and dual solution for (6). 

The Lagrangian for (6) is: 
\begin{equation}
L(\mathbf{w}, b, \mathbf{\alpha} ) = \frac{1}{2} ||w||^{2} - \sum_{i} \alpha_i [ y_i (\mathbf{w}^{T}\phi(x_i)+b) - 1 ] 
\tag{8}
\end{equation}

The stationarity is: 
\begin{equation}
\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} -  \sum_{i} \alpha_i y_i \phi(x_i) = 0 \Rightarrow  \mathbf{w} = \sum_{i} \alpha_i y_i \phi(x_i)
\tag{9}
\end{equation}

\begin{equation}
\frac{\partial L}{\partial b} = 0 -  \sum_{i} \alpha_i y_i  = 0 \Rightarrow   \sum_{i} \alpha_i y_i = 0 
\tag{10}
\end{equation}

Substituting (9) and (10) into (8) gives us: 

\begin{equation}
L = \frac{1}{2} \mathbf{w}^T  \mathbf{w} - \sum_{i} \alpha_i [ y_i (\mathbf{w}^{T} \phi(x_i)+b) - 1] 
\end{equation}

\begin{equation}
= \frac{1}{2}  \langle \sum_{i} \alpha_i y_i \phi(x_i) , \sum_{j} \alpha_j y_j \phi(x_j) \rangle  - \sum_{i} \alpha_i [ y_i (\mathbf{w}^{T} \phi(x_i)+b)] + \sum_i \alpha_i 
\end{equation}

\begin{equation}
= \frac{1}{2} \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j  \phi(x_i)^T  \phi(x_j) - \sum_{i} \alpha_i y_i \mathbf{w}^{T} \phi(x_i) - b \sum_{i} \alpha_i  y_i + \sum_i \alpha_i
\end{equation}

\begin{equation}
= \frac{1}{2}  \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j  \phi(x_i)^T  \phi(x_j) - \sum_{i} \alpha_i y_i ( \sum_{j} \alpha_j y_j \phi(x_j))^T\phi(x_i) + \sum_i \alpha_i 
\end{equation}

\begin{equation}
= \frac{1}{2}  \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j  \phi(x_i)^T  \phi(x_j) - \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j  \phi(x_i)^T  \phi(x_j)  + \sum_i \alpha_i  
\end{equation}

\begin{equation}
=  \sum_i \alpha_i  - \frac{1}{2}  \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j  \phi(x_i)^T \phi(x_j) 
\end{equation}

subject to its Dual feasibility: 
\begin{equation}
 \alpha_i \geq 0
  \tag{11}
\end{equation} 
 and (10)

Now we can write the dual form of (8):
\begin{equation}
L(\mathbf{\alpha} ) = \sum_i \alpha_i  - \frac{1}{2}  \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j  \phi(x_i)^T \phi(x_j), \\ 
s.t.,  \alpha_i \geq 0,  \sum_{i} \alpha_i y_i = 0
\tag{12}
\end{equation}

$\mathbf{\alpha}$ can be learnt by the SMO algorithm [2]. Once $\mathbf{\alpha}$ is learnt from the dual form, we can use $\mathbf{\alpha}$ to compute $\mathbf{w}$ and $b$ according to  (9) and (10). 

Note that its Primal feasibility indicates that: 
\begin{equation}
  y_i (\mathbf{w}^{T}\phi(x_i)+b) - 1  \geq 0,  i = 1,...,k 
 \tag{13}
\end{equation} 

And its Complementary shows: 
\begin{equation}
 \alpha_i [ y_i (\mathbf{w}^{T}\phi(x_i)+b) - 1 ]  = 0  
 \tag{14}
\end{equation} 

(11), (13) and (14) suggest that for every training sample, either $\alpha_i = 0 $ or $y_i (\mathbf{w}^{T}\phi(x_i)+b) =1$. The training samples whose associated $\alpha_i$ are considered as support vectors. 

Classification
-------------
For classification each sample $x^{\*}$ is plugged into the hyper plane defined in (1).  By substituting (9) into (1) we get the estimated $y^{\*}$: 
\begin{equation}
 y^* = \sum_i^N \alpha_i y_i \phi(x_i)^T \phi(x^*) +b 
 \tag{15}
\end{equation}
where N is the total number of training samples. 

Since only the $\alpha_i$ for support vectors are not zero,  the estimated $y^*$ depends only on those support vectors. 


References 
---------------
[1] [Convex Optimization: Spring 2015](http://www.stat.cmu.edu/~ryantibs/convexopt-S15/)  
[2] Platt, J. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines.