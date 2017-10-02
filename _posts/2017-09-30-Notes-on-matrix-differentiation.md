---
layout: post
title: "Notes on Matrix Differentiation"
description: ""
category: Maths 
tags: [Matrix]
---


In this post, the derivatives of matrices will be discussed for various cases. Through out the post, the following notations are used. 

 - $X$ = [$x_{ij}$]: a $M \times N$ matrix  
 - $X_{ik}$: the element of the matrix $X$ which is indexed by the $i$-th row and the $k$-th column 

Matrix-Valued Derivatives
-----------------------

----------
__Differentiating a scalar function regarding a matrix__ 
 
The derivative of a function $f$ of a scalar variable $x$ with respect to a matrix $X \in R^{M \times N}$ can be written as a matrix of $f$ differentiated w.r.t each element $X_{ij}$, namely:
 $$ \frac{\partial f}{\partial X} = 
\left[ \frac{\partial f}{\partial X_{ij}} \right] =
  \left[ 
  {\begin{array}{cccc}
   \frac{\partial f}{\partial X_{11}}, & \frac{\partial f}{\partial X_{12}} &,..., &\frac{\partial f}{\partial X_{1N}} \\ 
  \frac{\partial f}{\partial X_{21}}, & \frac{\partial f}{\partial X_{22}} 
  &,..., &\frac{\partial f}{\partial X_{2N}} \\
  ..., & ..., &  ..., &  ... \\
  \frac{\partial f}{\partial X_{M1}}, & \frac{\partial f}{\partial X_{M2}} 
  &,..., &\frac{\partial f}{\partial X_{MN}} \\
  \end{array} } 
  \right] \in R^{M \times N} $$

----------
__Differentiating a function matrix regarding a scalar__
 
When $X_{ij}$ is a function of a scalar variable $x$, the derivative of $X$
w.r.t. $x$ is defined as the matrix of each element $X_{ij}$ differentiated w.r.t $x$, namely 
 $$
  \frac{\partial X}{\partial x} = 
\left[ \frac{\partial X_{ij}}{\partial x} \right] = 
  \left[ {\begin{array}{cccc}
   \frac{\partial X_{11}}{\partial x}, & \frac{\partial X_{12}}{\partial x} &,..., &\frac{\partial X_{1N}}{\partial x} \\
   \frac{\partial X_{21}}{\partial x}, & \frac{\partial X_{22}}{\partial x} &,..., &\frac{\partial X_{2N}}{\partial x} \\
  ..., & ..., &  ..., &  ... \\
     \frac{\partial X_{M1}}{\partial x}, & \frac{\partial X_{M2}}{\partial x} &,..., &\frac{\partial X_{MN}}{\partial x} \\
  \end{array} } \right] \in R^{M \times N} $$


Expanding a matrix element 
-----------------------
It is useful to expand each element of matrix $X_{ik}$ into sum of multiplications when $X$ is a matrix product.

For example, $X_{ik}$ can be written as $$X_{ik} = [AB]_{ik} = \sum A_{ij} B_{jk}, \tag{1} $$ when $X= AB$.

Likewise, $[ABC]_{il}$ can be expanded as $$[ABC]_{il} = \sum_j A_{ij}[BC]_{jl} = \sum_j A_{ij} \sum_{k}B_{jk}C_{kl} $$

Derivatives of an Inverse
-----------------------
Since $XX^{-1}  = I$, we can differentiate both sides w.r.t $x$ to obtain the following:
$\frac{\partial X}{\partial x} X^{-1} + X \frac{\partial X^{-1}}{\partial x} = 0 \tag{2} $

(2) can be transformed into: 

$\frac{\partial X^{-1}}{\partial x} = -X^{-1} \frac{\partial X}{\partial x} X^{-1} \tag{3}$

For $x=X_{ij}$,

$\frac{\partial X^{-1}}{\partial x} 
= -X^{-1} \frac{\partial X}{\partial X_{ij}} X^{-1} = -X^{-1} I_{ij} X^{-1}, \tag{4}$

where $I_{ij}$ is a matrix with 1 set to the $(i,j)$ component, and 0 to all other components.

Derivatives of a Trace 
-----------------------
The [trace](https://en.wikipedia.org/wiki/Trace_%28linear_algebra%29) of a n-by-n square matrix is defined as the sum of its elements on the main diagonal. For example, 
$tr(X) = \sum_{i} X_{ii} \tag{5}$  
We will use (1) and (5) in deriving the derivatives of traces. 


----------
__Firt-order Derivatives__

Let $f$ be $tr[AXB]$, we can write 

$$f = \sum_{i} [AXB]_{ii} = \sum_{i} \sum_{j} A_{ij}[XB]_{ji} = 
\sum_{i} \sum_{j} A_{ij} \sum_{k} X_{jk} B_{ki} 
= \sum_{i} \sum_{j} \sum_{k} A_{ij} X_{jk} B_{ki}$$

Take the derivative of $f$ w.r.t $X_{jk}$, and we can get:
$\frac{\partial f}{\partial X_{jk}} = \sum_i A_{ij} B_{ki} = \sum_i B_{ki} A_{ij}  = [BA]_{kj} \tag{6}$

In (6), as the order $(j,k)$ index in $X_{jk}$ is different from the $(k,j)$ index in $[BA]_{kj}$, we have to transpose the results in (6) to obtain its matrix form.

This gives us: 
$\frac{\partial f}{\partial X} = (BA)^{T} = A^TB^T$, i.e., $\frac{\partial tr[AXB]}{\partial X} = A^TB^T$

Similarly, let $f$ be $tr[AX^TB]$, then 

$$f = \sum_{i} [AX^TB]_{ii} = \sum_{i} \sum_{j} A_{ij}[X^TB]_{ji} $$
$$= \sum_{i} \sum_{j} A_{ij} \sum_{k} X^{T}_{jk} B_{ki} $$
$$= \sum_{i} \sum_{j} A_{ij} \sum_{k} X_{kj} B_{ki} $$
$$= \sum_{i} \sum_{j} \sum_{k} A_{ij} X_{kj} B_{ki}$$.

Take the derivative of $f$ w.r.t $X_{kj}$, and we can get:
$\frac{\partial f}{\partial X_{kj}} = \sum_i A_{ij} B_{ki} = \sum_i B_{ki} A_{ij}  = [BA]_{kj} \tag{7}$

In (7), Since the order $(k,j)$ index in $X_{kj}$ is the same as  $(k,j)$ index in $[BA]_{kj}$, no matrix transposing is required to obtain its matrix form. 

As such, $\frac{\partial f}{\partial X} = (BA)$, i.e., $\frac{\partial tr[AX^TB]}{\partial X} = BA$


----------
__Higher-order Derivatives__

Let $f$ be $tr[AXBXC^T]$,

$$f = \sum_i [AXBXC^T]_{ii}$$ 
$$= \sum_i \sum_j \sum_k \sum_l \sum_m A_{ij}X_{jk}B_{kl}X_{lm} C^{T}_{mi}$$
$$= \sum_i \sum_j \sum_k \sum_l \sum_m A_{ij}X_{jk}B_{kl}X_{lm} C_{im} $$

Note that there are two $X$'s in the formula above, so we need to take derivative of $f$ w.r.t $X_{lm}$ and $X_{jk}$.

$\frac{\partial f}{\partial X_{jk}} 
= \sum_i \sum_l \sum_m A_{ij}B_{kl}X_{lm}C_{im} 
= \sum_i \sum_l \sum_m B_{kl}X_{lm}C^T_{mi}A_{ij}  
= [BXC^TA]_{kj} \tag{8}$

$\frac{\partial f}{\partial X_{lm}} 
= \sum_i \sum_j \sum_k A_{ij} X_{jk} B_{kl} C_{im}
= \sum_i \sum_j \sum_k C^T_{mi} A_{ij} X_{jk} B_{kl}
= [C^TAXB]_{ml} \tag{9}$

Deriving (8) an (9) requires some observations. Take (8) as an example, since the indices $i,l$ and $m$ are summed, only indices $k$ and $j$ can appear in the matrix form. That is why $A_{ij}B_{kl}X_{lm}C_{im}$ are arranged as $B_{kl}X_{lm}C^T_{mi}A_{ij}$ to ensure only $k$ and $j$ are left after summing $i,l,$ and $m$.   

For (8), since the indices ($k,j$) in $$[BXC^TA]_{kj}$$ are in different order from the indices ($j,k$) in $X_{jk}$, a matrix transposition is required. That is also the case for (9). 

Putting (8) and (9) together gives us:

$\frac{\partial f}{\partial X} 
=  \left[ \frac{\partial f}{\partial X_{jk}} \right] + \left[ \frac{\partial f}{\partial X_{lm}} \right]
= (BXC^TA)^T + (C^TAXB)^T 
= A^TCX^TB^T + B^TX^TA^TC $,

That is, $\frac{tr[AXBXC^T]}{\partial X} = A^TCX^TB^T + B^TX^TA^TC $

Alternatively, we can apply the rules we have learnt from doing first-order derivatives, i.e., 

$\frac{\partial tr[AXB]}{\partial X} = A^TB^T$, to derive $\frac{tr[AXBXC^T]}{\partial X}$

$$\frac{tr[AXBXC^T]}{\partial X} \\ $$ 
$$=\frac{tr[(A)(X)(BXC^T)]}{\partial X}+\frac{tr[(AXB)X(C^T)]}{\partial X}$$ (Product rule for derivatives)  
$$= A^T(BXC^T)^T + (AXB)^T(C^T)^T \\ $$ 
$$= A^TCX^TB^T + B^TX^TA^TC$$


References
----------------------- 
[1] Petersen, K. B., & Pedersen, M. S. (2008). The matrix cookbook. Technical University of Denmark, 7, 15.  
[2] Johannes Traa [Matrix Calculus - Notes on the Derivative of a Trace](cal.cs.illinois.edu/~johannes/research/matrix%20calculus.pdf)
