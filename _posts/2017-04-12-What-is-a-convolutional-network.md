---
layout: post
title: "What is a convolutional neural network"
description: ""
category: Machine Learning 
tags: [Neural networks]
---


This post is adapted from my answer to the question "What is a convolutional neural network? " on **Quora**.
 
Neural Networks receive an input, and transform it via a series of hidden layers.
Convolutional neural networks (CNNs) are a type of neural networks which explicitly assume that the inputs are images.  They encodes certain properties of image data into their architecture, which improves generalization performance through reducing the number of network parameters.
 
Motivation
------------------
A ordinary neural network  or multi-layer perceptron  (MLP) is made up of input/out layers and a series of hidden layers. Fig. 1 shows an example of MLP which contains only one hidden layer.  Note that each hidden layer is fully connected to the previous layer, i.e., each neuron in the hidden layer connects to all neurons in the previous layer. 
 
<div>
<center>
<img  alt="A multi-layer perceptron" src="/img/2017-04-12/multi-layer-perceptron.png">
</center>
<center>Figure 1. A multi-layer perceptron [3] </center>
</div>
 
There are three main problems if we use MLP for image recognition tasks:
 
- MLPs do not scale well
The generalization performance of the MLP will be eclipsed by its excessive free parameters, i.e., weights.  For example, each image from the ImageNet [4] has a size of 256x256x3. That means each neuron in the hidden layer  will have 256x256x3=196608 connections with each pixel in the input image in Fig. 1.  The total number of weights would scale up quickly if we want to add more neurons or hidden layers. The enormous number of weights(parameters) produced by full connectivity would quickly lead to over-fitting.
               
 - MLPs ignore pixel correlation
It is an important property of images that nearby pixels are more strongly correlated than distant pixels. This property is not taken advantage of by MLP due to the full connectivity.  This property suggests  local connectivity is preferred.
               
 - MLPs are not robust to image transformations
It is expected that the recognition algorithm should be robust to transformations applied to the input image. For example, for handwritten digit recognition, a particular digit should be assigned the same value regardless of its position within the image or of its size.
 
 For MLPs, any subtle change in scale or position from the input layer would produce significant changes in following layers. It is therefore advantageous to incorporate into the network some invariance to common changes that could occur in images, e.g., translation,  scale, etc.
               
Accordingly image recognition tasks require a new type of neural network.
               

Network Design Strategy
------------------
 
The design of CNNs follows the guidelines presented below [1]: 
 
Two performance measures should be considered when evaluating a neural network, namely the learning speed and the generalization performance.  Generalization is the first property that should be considered as it determines the amount of training data the network needs to produce correct response on unseen data.
 
LeCun wrote in [1] that "good generalization performance can be obtained if some prior knowledge about the task is built into the network".  Tailoring the network architecture affects the network complexity. The network complexity is reflected on the number of its free parameters, which should be minimized to increase the likelihood of correct generalization.  However the work must be done without deteriorating the network's capability to compute the desire function.
 

Convolutional Neural Networks
------------------
 
To utilize the prior knowledge on image recognition,  CNNs incorporate the following concepts into the design:
 
- Local connectivity  
  Local connectivity is a solution to the over-parameterization problem.  The advantage of using local features and the derived high order features has been demonstrated in classical work in visual recognition.  This knowledge can be easily built into the network, by forcing the neurons to receive only local information.
  This notion is illustrated in  Fig. 2 where layer m-1 can be considered as input images, and layer m is the hidden layer.  It can be seen from the figure that each neuron in the hidden layer connects to only 3  adjacent input pixels in the image.  Local connectivity also takes advantage of the pixel correlation from input image, as each neuron in the hidden layer only cares about nearby pixels in a neighbourhood.  
             
<div>
<center>
<img  alt="A neural network with local connectivity" src="/img/2017-04-12/local-connectivity.png">
</center>
<center>Figure 2. A neural network with local connectivity [5] </center>
<br>
</div>
                
 - Weight sharing  
   One problem of image recognition tasks is that images that contain the same semantic object could have the object at various locations on the image.  As a result,  classical work in visual recognition detects local features at various location on the input image. Weight sharing is a solution that stimulates the approach of applying local feature detectors at different positions of the image. It is also a solution to improve network robustness against image transformations.
   As depicted in Figure 3,  the three neurons in hidden layer m share the same weight.  Weights of the same colour are constrained to be identical.  Note that the hidden layer m could contain multiple planes of neurons that share the same weights.  These planes are referred to as "feature maps".  Neurons in every feature map are constrained to have identical weights, which is equivalent to performing  the same operation in different parts of the image.
   Since each neuron is only connected to part of an image, and keeps the information in the corresponding feature map. This behaviour is equivalent to a convolution with a small size kernel, followed by a squashing function [2]. This is also how CNN was named.  
                   
<div>
<center>
<img  alt="A neural network with weight sharing" src="/img/2017-04-12/weight-sharing.png">
</center>
<center>Figure 3.  A neural network with weight sharing [5]  </center>
<br>
</div>
               
 -  Sub sampling
    As an object can appear at various locations on the input image, the precise location of a local feature is not important to the classification. That means the network can afford to lose the exact position information. However an approximate position information should be retained so that the next layer can possibly detect higher- order feature.  As a result the feature maps need not be as the same size of the input image. Instead, a feature map can be built by sub sampling the input image. Sub sampling provides a form of invariance to distortions and translations.
               
Piecing these concepts together results in LeNet, one of the first CNN, which is illustrated in Fig 4 [5]. Although Fig 4 does not precisely reflect the architecture presented in [2],  it can still be used as an meaningful illustration. 
 
As shown in Fig 4, LeNet is made up of alternating convolutional and sub-sampling layers in its lower-layers. Its upper layers correspond to fully-connected layers employed in MLPs.   


 <div>
 <center>
 <img  alt="The LeNet" src="/img/2017-04-12/LeNet-architecture.png">
 </center>
 <center>Figure 4.  The LeNet [5]   </center>
 <br>
</div>
 
 
LeNet has been shown to generalize well compared to MLP in [2].  This network was the origin of much of the recent CNN architectures.
 

References
------------------
[1] LeCun, Yann. "Generalization and network design strategies." Connectionism in perspective (1989): 143-155.  
[2] Le Cun, B. Boser, et al. "Handwritten digit recognition with a back-propagation network." Advances in neural information processing systems. 1990.  
[3] Multilayer Perceptron http://deeplearning.net/tutorial/mlp.html  
[4]Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database" Computer Vision and Pattern Recognition, 2009.  
[5] Convolutional Neural Networks (LeNet) http://deeplearning.net/tutorial/lenet.html
 
 