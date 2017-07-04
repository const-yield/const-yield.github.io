---
layout: post
title: "How the Backpropagation algorithm works"
description: ""
category: Machine Learning 
tags: [Neural networks]
---

How the Backpropagation algorithm works
==
 
This blog post is my study note on [chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html) of the book "**Neural Networks and Deep Learning**".  The derivation of the four fundamental equations is the focus of this post.
 
Notations
------------------
 
The following notations are used in the explanation, in reference to  Figure 1.
 
- $w_{j,k}^{l}$ is the weight of the $k^{th}$ neuron in layer ($l$ -$1$) to the $j^{th}$ neuron in layer $l$
 
-  $b_{j}^{l}$ is the bias of the $j^{th}$ neuron in layer $l$ 
 
- $z_{j}^{l}$ is the weighted input to the $j^{th}$ neuron in layer $l$, i.e., $z_{j}^{l} = \sum_{k} w_{j,k}^{l}* a_{k}^{l-1}  + b_{j}^{l}$，where $ a_{k}^{l-1}$ is the activation of the $j^{th}$ neuron in layer $l$-$1$
 
- $a_{j}^{l}$ is the activation of the $j^{th}$ neuron in layer *l*. It is the response of the $\sigma$ function given the weighted input to the $j^{th}$ neuron in layer *l*, i.e., $a_{j}^{l} = \sigma(z_{j}^{l}) = \sigma( \sum_{k} w_{j,k}^{l}* a_{k}^{l-1}  + b_{j}^{l})$
 
- $\delta_{j}^{l}$  is the error in the $j^{th}$ neuron in layer $l$ computed in the back propagation algorithm
 
They can also be expressed in matrix form below:
 
- $w^{l}$ is a n-by-m weight matrix for neurons in layer $l$
 
-  $b^{l}$ is an n-by-1 bias vector for neurons in layer $l$
 
- $z^l$ is a m-by-1 vector of weighted input to the neurons in layer $l$. It can be formulated as $z^l \equiv w^l * a^{l-1} + b^{l}$ , where $a^{l-1} $ is the activation vector of layer $l$-$1$
 
-  $a^{l}$ is a m-by-1 activation vector of layer *l*, which can be formulated as $a^{l} = \sigma(z^l) = \sigma( w^{l}* a^{l-1} +b^{l})$
 
- $\delta^{l}$ is a m-by-1 error vector for neurons in layer $l$
 
The notation of [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_%28matrices%29) is used in the equations:
 
- $s \odot t$ is the element-wise product of two vectors $s$ and $t$.
 
<div>
<center>
<img  src="/img/2017-06-29/notations.png">
</center>
<center>Figure 1.  The notations used in Back-propagation algorithm</center>
<br>
</div>
 
The big picture
------------------
The Backpropagation algorithm describes how changes on the parameters $w$ and $b$ affect the cost function $C$. This is realized by computing the partial derivatives of  $C$  w.r.t. parameters $w$ and $b$ in the network, namely $\frac{\partial C}{\partial  w_{j,k}^{l}}$, and $\frac{\partial C}{\partial  b_{j}^{l}}$.
 

Assumptions
------------------

The partial derivatives of  $C$  w.r.t. parameters are computed based on the following assumptions:
 
- The cost function $C$ can be written as an average $C = \frac{1}{n} \sum_{x} C_x$ over the cost functions $C_x$ for individual training samples
 
- The cost function $C$ can be written as a function of outputs from the neural network
 
 
 
Four fundamental equations
------------------ 
 
Before diving into these four equations, we need to define $\delta_{j}^{l}$.  Apparently $\delta_{j}^{l}$ is affected by changing the cost $C$, and the corresponding input $z_{j}^{l}$. As such $\delta_{j}^{l}$  is defined as the change rate of $C$ w.r.t. the input to the $j^{th}$ neuron in layer $l$ , i.e., $\delta_{j}^{l} \equiv \frac{\partial C}{\partial  z_{j}^{l}} $.
 
The derivation of the four equations follow such a pattern:
 
1. define the formula for the partial derivative;
2. identify associated variables from existing equations;
 3. expand the formula for the partial derivative by incorporating the associated variables using the chain-rule;
 4. simplify the expanded partial derivative using existing equations.
 
BP1
-

**BP1** is an equation measures how fast the cost $C$ is changing as a function of  the error of the $j^{th}$ neuron in the output layer $L$, i.e., $\frac{\partial C}{\partial  z_{j}^{L}}$.
 
Note that $a_{j}^{l} = \sigma( z_{j}^{l}) = \sigma( \sum_{k} w_{j,k}^{l}* a_{k}^{l-1} + b_{j}^{l})$; we can insert $a_{j}^{L}$ to $\frac{\partial C}{\partial  z_{j}^{L}}$ according to the chain rule:
 
\begin{equation}
   \delta_{j}^{L}  \equiv \frac{\partial C}{\partial  z_{j}^{L}} =  \frac{\partial C}{\partial  a_{j}^{L}} * \frac{\partial a_{j}^{L}}{\partial z_{j}^{L}} =  \frac{\partial C}{\partial  a_{j}^{L}} *  \sigma'(z_{j}^{L}) 
   \tag{1}
\end{equation}

From (1) we can see that the multiplication of $\frac{\partial C}{\partial  a_{j}^{L}}$ and $\sigma'(z_{j}^{L})$ are indexed by the neuron index $j$.
 
The matrix form of **BP1** can be formulated as:
 
\begin{equation}
   \delta^{L} = \frac{\partial C}{\partial  a^{L}} \odot \sigma'(z^{L}) = \nabla_{a}^{C} \odot \sigma'(z^{L}) 
   \tag{2}
\end{equation}
 
BP2
-- 
 
**BP2** is an equation for expressing the error $\delta_{j}^{l}$ in layer $l$ in terms of the error $\delta_{j}^{l+1}$ from the next layer $l$+$1$.
 
By definition, $\delta_{j}^{l} \equiv \frac{\partial C}{\partial  z_{j}^{l}}$. Assuming that we know the error in layer $l$+$1$, i.e., $\delta_{j}^{l+1} \equiv \frac{\partial C}{\partial  z_{j}^{l+1}}$,   we can introduce  $z_{j}^{l+1}$ to $\frac{\partial C}{\partial  z_{j}^{l}}$ using the chain rule, in order to expand $\frac{\partial C}{\partial  z_{j}^{l}}$. Note that $z_{j}^{l+1}$ actually affects all neurons in layer $l$ due to full connectivity. In other words,  $z_{j}^{l}$ is influenced by all  $z_{j}^{l+1}$ in layer $l$+$1$. As a result, the expansion of $\frac{\partial C}{\partial  z_{j}^{l}}$ can be written as follows:
 
\begin{equation}
  \delta_{j}^{l} \equiv \frac{\partial C}{\partial  z_{j}^{l}} = \sum_{k} \frac{\partial C}{\partial  z_{k}^{l+1}} * \frac{\partial z_{k}^{l+1}}{\partial  z_{j}^{l}} = \sum_{k} \delta_{k}^{l+1}* \frac{\partial z_{k}^{l+1}}{\partial  z_{j}^{l}} $ 
  \tag{3}
\end{equation}
 
Since $z_{k}^{l+1} = \sum_{j} w_{k,j}^{l+1}* \sigma(z_{j}^{l})  + b_{k}^{l+1}$, we have:

\begin{equation}
 \frac{\partial z_{k}^{l+1}}{\partial  z_{j}^{l}} =  w_{k,j}^{l+1}* \sigma'(z_{j}^{l}) 
   \tag{4}
\end{equation}
 
Combining (3) and (4) gives us **BP2**:  $\delta_{j}^{l} = \sum_{k} w_{k,j}^{l+1} \sigma'(z_{j}^{l}) \delta_{k}^{l+1}$, where $j$ and $k$ index the neurons in layers $l$ and $l$+$1$ respectively.
 
**BP2** can be interpreted as follows: each neuron in layer $l$+$1$ contributes its weighted error, i.e., $w_{k,j}^{l+1}*\delta_{k}^{l+1}$ to the error of the $j^{th}$ neuron in layer $l$. The sum of the weighted error, i.e.,  $\sum_{k} w_{k,j}^{l+1}\delta_{k}^{l+1}$ is then scaled by $\sigma'(z_{j}^{l})$ to result in the error for the $j^{th}$ neuron in layer $l$, i.e., $\delta_{j}^{l}$.
 
The matrix form of **BP2** can be formulated as: $\delta^{l-1} = (W^{l})^{T} \delta^{l} \odot  \sigma'(z^{l-1}) $,
where:
 
- $\delta^{l-1}$ is a m-by-1 vector
 
-  $W^{l}$ is a n-by-m matrix
 
- $\delta^{l}$ is a n-by-1 vector
 
- $\sigma'(z^{l-1})$ is a m-by-1 vector
 
 
BP3
--
 
**BP3** is an equation that measures how fast the cost $C$ is changing as a function of the error of any bias values in the network, i.e., $\frac{\partial C}{\partial  b_{j}^{l}}$.
 
Note that $z_{j}^{l} = \sum_{k} w_{j,k}^{l}* \sigma(z_{k}^{l-1})  + b_{j}^{l}$;  we can introduce $z_{j}^{l}$ to  $\frac{\partial C}{\partial  b_{j}^{l}}$ based on the chain rule:


\begin{equation}
\frac{\partial C}{\partial  b_{j}^{l}} =  \frac{\partial C}{\partial  z_{j}^{l}} * \frac{\partial z_{j}^{l}}{\partial  b_{j}^{l}} = \delta_{j}^{l} * \frac{\partial z_{j}^{l}}{\partial  b_{j}^{l}} $   
   \tag{5}
\end{equation}
 
 
Since $z_{j}^{l} = \sum_{k} w_{j,k}^{l}* \sigma(z_{k}^{l-1})  + b_{j}^{l}$, we have:

\begin{equation}
\frac{\partial z_{j}^{l}}{\partial  b_{j}^{l}} = 1$ 
   \tag{6}
\end{equation} 
 
Combining (5) and (6),  we have **BP3** :  $\frac{\partial C}{\partial  b_{j}^{l}} = \delta_{j}^{l} $
 
The matrix form of **BP3** can be expressed as $\frac{\partial C}{\partial  b^{l}} = \delta^{l}$, where $b^{l}$ and $\delta^{l}$ are both m-by-1 vectors.
 
 
BP4
-- 
 
**BP4** is an equation that reveals how fast the cost $C$ is changing as a function of any weight value in the network, i.e., $\frac{\partial C}{\partial  w_{j,k}^{l}}$.
 
Note that $a_{j}^{l} = \sigma( z_{j}^{l}) = \sigma( \sum_{k} w_{j,k}^{l}* a_{k}^{l-1} + b_{j}^{l})$; we can insert $a_{j}^{l}$ and $z_{j}^{l}$ to $\frac{\partial C}{\partial  w_{j,k}^{l}}$ based on the chain rule:
 
\begin{equation}
\frac{\partial C}{\partial  w_{j,k}^{l}} = \frac{\partial C}{\partial  a_{j}^{l}} * \frac{\partial a_{j}^{l}}{\partial  z_{j}^{l}} * \frac{\partial z_{j}^{l}}{\partial  w_{j,k}^{l}} = \frac{\partial C}{\partial  z_{j}^{l}} * \frac{\partial z_{j}^{l}}{\partial  w_{j,k}^{l}}  
\tag{7}
\end{equation}
 
 
Since $\delta_{j}^{l} \equiv \frac{\partial C}{\partial  z_{j}^{l}}$ (**BP1**) ,  we have:

\begin{equation}
  \frac{\partial C}{\partial  z_{j}^{l}} = \delta_{j}^{l} 
  \tag{8}
  \end{equation}
 
  
Due to  $z^{l}_{j} =\sum_{k} w_{j,k}^{l}* a_{k}^{l-1} + b_{j}^{l}$, we have:  

\begin{equation}
\frac{\partial z_{j}^{l}}{\partial  w_{j,k}^{l}} =  a_{k}^{l-1} 
  \tag{9}
\end{equation}
  
Substituting (8) and (9) into (7) results in **BP4**:  $\frac{\partial C}{\partial  w_{j,k}^{l}} = a_{k}^{l-1} * \delta_{j}^{l}$.
 
BP4 can be written as a less index-heavy form: $\frac{\partial C}{\partial  w} = a_{in} * \delta_{out}$, where:
 
- $a_{in}$ is the activation of neuron input to weight $w$
 
- $\delta_{out}$ is the error of the neuron output from weight $w$
 
From it we can see that, when $a_{in} \approx 0$, $\frac{\partial C}{\partial  w}$ is very small, which indicates that low-activated neurons learn slowly.
 
 
Summary
--
Here is the summary of the four fundamental equations:
 
- $\delta_{j}^{L}  = \frac{\partial C}{\partial  a_{j}^{L}} *  \sigma'(z_{j}^{L}) $ (BP1)
 - $\delta_{j}^{l} = \sum_{k} w_{k,j}^{l+1} \sigma'(z_{j}^{l}) \delta_{k}^{l+1}$ (BP2)
 
 - $\frac{\partial C}{\partial  b_{j}^{l}} = \delta_{j}^{l} $ (BP3)
 
 - $\frac{\partial C}{\partial  w_{j,k}^{l}} = a_{k}^{l-1} * \delta_{j}^{l}$. (BP4)
 
and their simplified form:
 
- $\delta^{L} =  \nabla_{a}^{C} \odot \sigma'(z^{L}) $ (BP1)
 
- $\delta^{l-1} = (W^{l})^{T} \delta^{l} \odot  \sigma'(z^{l-1}) $ (BP2)
 
- $\frac{\partial C}{\partial  b^{l}} = \delta^{l}$ (BP3)
           
- $\frac{\partial C}{\partial  w} = a_{in} * \delta_{out}$ (BP4)
 

The Backpropagation algorithm
------------------ 
 
Bear in mind that the Backpropagation algorithm provides us with a way of computing the gradient of the $C$ regarding the parameters.  This is achieved by the use of the four fundamental equations:
 
- The output error of the network, i.e., $\delta^{L}$, can be calculated by **BP1**.
 - And then these errors are propagated from the last layer to previous layers using **BP2**. *That is the reason why the algorithm is called Backpropagation.*
 - Once the errors for all layers are available,  the desired partial derivatives can be determined by **BP3** and **BP4**.
 
The Backpropagation algorithm is summarized as follows:
 
1. Set the corresponding activation $a^{1}$ for the input layer, based on the given input $x$
2. Compute $z^l = w^l * a^{l-1} + b^{l}$, and $a^l = \sigma(z^l)$ for each $l$ = $2$,$3$,...,$L$
3. Compute the error at the last layer $\delta^{L}$ using **BP1**
 4. Compute the errors at previous layers $\delta^{l}$, where $l$ = $L$-$1$, $L$-$2$,...$2$ using  **BP2**
5.  Output the gradient of the cost function using **BP3** and **BP4**