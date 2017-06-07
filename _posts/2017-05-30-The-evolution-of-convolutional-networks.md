The Evolution of Convolutional Networks ImageNet
==
 
This blog post briefs the evolution of  Convoluational Neural Networks (CNN).  The selected CNNs were developed for the ImageNet Large-Scale Visual Recognition Challenges (ILSVRC).  They are reviewed according to their time of publications chronologically.
 
1989: LeNet
==
 
LeCun proposed LeNet (LeCun, 1989), one of the first CNNs, for the application of digit recognition.  The LeNet laid the foundation for Modern CNN models.  Some of  its characteristics, such as local connectivity,  weight sharing, and sub-sampling, are still shared by recent CNNs.  Detailed discussion of LeNet can be found in my previous [post](https://const-yield.github.io/2017-04-12-What-is-a-convolutional-network/)  "What is a convolutional neural network".
 
 
2012: AlexNet
==
 
After an incubation time of almost 20 years (from 1989 to 2012), the machine learning community witnessed the debut of AlexNet  (Krizhevsky, 2012) in 2012, a breakthrough for deep learning.  AlexNet was a CNN designed to recognize pictures in the ImageNet dataset, which contains over 15 million images in over 22,000 categories.
 
Recognizing such a large volume of object categories requires  a model with a large learning capacity.  CNNs are good candidate for such a learning task due to:
 
- The capacity of CNN models are controlled by their depth and breadth
 - They have strong and mostly correct assumptions about the nature of images
 - They have fewer connections than standard feed-forward neural networks
 
AlexNet was one of the largest convolutional neural network at the time of publication used in ILSVRC-2010 and ILSVRC-2012 competitions.  Its architecture consists of eight learned layers: five convolutional and three fully-connected layers (See Figure 1).
 
<div>
<center>
<img  src="/img/2017-05-30/alex_net_architecture.png">
</center>
<center>Figure 1.  The architecture of AlexNet (Krizhevsky, 2012) </center>
<br>
</div>
 
It was written with a highly-optimized GPU implementation, with a number of new features to improve its performance and reduce its training time:
 
-  Rectified Linear Units (ReLU) for faster training: the experiment results showed that networks with ReLUs consistently converge faster than those with saturating neurons. 
-  Training on multiple GPUs: the parallelization scheme puts half of the kernels on each of the two GPUs
-  Local Response normalization: though ReLUs have the property that they do not require input normalization to prevent them from saturating, local normalization was still found to aid generalization.
 
The following techniques were used in AlexNet to reduce over-fitting:
 
- Data augmentation -- artificially enlarge the data set using label-preserving transformations
- Dropout- randomly set zero to the output of each hidden neuron with a probability of 0.5
 
The AlexNet (single-CNN) obtained a top-5 error of 18.20% on the ILSVRC-2012 validation set.
 
 
2014: VGG Net
==
After the debut of AlexNet in 2012, a number of attempts had been made to tweak the structure of AlexNet to achieve better performance. The VGG network is a successful one of those attempts.
 
One prominent feature of the VGG net is that filters with very small receptive fields were employed throughout the whole network.  According to (Simonyan, 2014),  the
benefit of using small receptive field is two-fold:
 
- It makes adding more convolutional layers feasible due to its low computational cost
- Multiple convolutional Layers without spatial pooling in between has effectively bigger receptive field ( See [here](https://www.quora.com/What-is-a-CNN%E2%80%99s-receptive-field) for detailed explanation )
 
Another salient feature of the VGG net is its depth. In comparison with the AlexNet, it extended its depth to up to 19 layers.
 
The features of VGG's architecture are summarized below:
 
- Filters with very small 3-by-3 receptive field were used throughout the whole  net, which were convolved with stride 1.
- 1-by-1  convolutional filters were also incorporated to enhance the nonlinearity of the design function without affecting the receptive fields
- Spatial padding is such that the spatial resolution after convolution was preserved
- Spatial pooling is carried out by five max-pooling layers. 
- Max-pooling is performed over a 2-by-2 pixel window, with stride 2.
 
Details of its architecture can be found in Figure 2.
<div>
<center>
<img  src="/img/2017-05-30/vgg_net_architecture.png">
</center>
<center>Figure 2.  The architecture of VGG Net (Simonyan, 2014)  </center>
<br>
</div>
 
The VGG net (single net) obtained a top-5 error of 6.8% on the ILSVRC-2012 validation set.
 
2015: GoogLeNet
==
Increasing the size of a deep neural network is perhaps the most straightforward way of improving its performance.  This can be done by increasing the depth ( the number of network levels) or the width ( the number of units at each level). However the drawbacks of this simple solution comes with a larger number of parameters , and increased use of computational resources. 
 
GoogLeNet is the efficient network architecture proposed in (Szegedy, 2015). Its goal is to keep a computational budget of 1.5 million multiply-adds at inference time, so that the networks can be put into practical use at a reasonable cost.  A new  level of network organization named "Inception module" was introduced into the architecture to enable GoogLeNet to "go deeper with convolution".
 
The Inception architecture considers how an optimal local sparse structure of a convolutional vision network can be approximated and covered by readily available dense components.  Two types of Inception architectures were introduced, namely the na?ve version, and a dimention-reduced version.
 
The na?ve Inception architecture (shown in Figure 3) has the following features :
- As suggested by (Arora, 2014), a  layer-by-layer construction is employed, where one should analyze the correlation statistics of the last layer and cluster them into groups of units with high correlation. These clusters form the units of next layer and are connected to units in the previous layer.
- It is assumed that each unit from an earlier units  corresponds to some region of the input image and these units are grouped into filter banks. As a result, lots of clusters will concentrate in a single region and they can be covered by a layer of 1-by-1 convolution.
-  To avoid patch-alignment issues,  the implementation of  current Inception architecture is restricted to 1-by-1, 3-by-3 and 5-by-5 filter sizes
-  Due to the prevalence of the  pooling operations are in convolutional networks, the Inception architecture adds an alternative parallel pooling path in each stage for additional benefits
 
<div>
<center>
<img  src="/img/2017-05-30/google_net_architecture_01.png">
</center>
<center>Figure 3.  The na?ve Inception architecture (Szegedy, 2015) </center>
<br>
</div>
 
However this na?ve Inception module has excessive number of outputs due to the merge of pooling and convolutional outputs.  To tackle this problem, 1-by-1 convolutions are used to reduce the output dimension before performing the expensive 3-by-3 and 5-by-5 convolutions (See Figure 4).
 
<div>
<center>
<img  src="/img/2017-05-30/google_net_architecture_02.png">
</center>
<center>Figure 4.  The Improved Inception architecture (Szegedy, 2015) </center>
<br>
</div>
 
The GoogLeNet is a 22-layer-deep network which deploys the Inception modules at higher layers, while keeping lower layers in traditional convolutional manner.  This is for obtaining memory efficiency during training but not necessary.  GoogLeNet employs only 5 million parameters, which is only 1/12 of the number of parameters in its predecessor AlexNet. The architecture of GoogLeNet is described in Figure 5.
 
<div>
<center>
<img  src="/img/2017-05-30/google_net_architecture_03.png">
</center>
<center>Figure 5.  The GoogLeNet architecture (Szegedy, 2015) </center>
<br>
</div>
 
The GoogLeNet obtained a top-5 error of 6.67% on the ILSVRC-2014 validation set, ranking first among other participants.
 
 
2015: BN-Inception
==
 
Stochastic Gradient Descent (SGD) has proved to be a simple and effective means of training deep networks,  however it requires careful tuning of the hyper-parameters for the model,  especially the learning rates, and initialization values.   As the input to each layer is affected by  the parameters of preceding network layers,  the distribution of layers' input changes from one layer to another.  This has caused issues to the training  as the layers need to continuously adapt to new distributions.   Having a fixed distribution across layers makes training more efficient, as the parameters do not need to readjust to compensate for the change in the distribution.
 
The change in the distributions of internal nodes of a deep network is referred to as ***Internal Covariance Shift.*** A new mechanism called ***Batch Normalization*** is proposed to reduce internal covariance shift. This is done by a normalization step that fixes the means and variances of layer inputs.  Experiment results show that Batch normalization allows higher learning rates without the risk of divergence, and it regularizes the model and reduces the need for dropout.
 
The BN-Inception model reached a top-5 error of 4.9% on the ILSVRC-2012 validation set.
 
 
2015: The Highway networks
=
 
Network depth, i.e., the number of successive computational layers,  has played an important role in the success of convolutional networks.  This is due to that deep networks can represent certain function classes much more efficiently than the shallow ones.  However as the network depth increases, the training of the network also gets difficult.
 
Inspired by the skip connections between layers used in neural networks, Srivastava proposed the Highway Networks in (Srivastava, 2015). The Highway Networks modified the architecture of very deep feed-forward networks to make it easier to flow information across layers.  This was realized through an LSTM-inspired adaptive gating mechanism.
 
The experiment results in Figure 6. show that Highway network achieves lower training accuracy with deeper networks.
 
<div>
<center>
<img  src="/img/2017-05-30/highway_network_01.png">
</center>
<center>Figure 6.  Comparison of optimization of plain networks and highway networks of various depths. (Srivastava, 2015) </center>
<br>
</div>
 
 
2016: Inception V2 and V3
==
 
In comparison with VGG net, the computational cost of Inception is much lower. However the complexity of the Inception architecture    hinders the changes to the network.  It makes it difficult to adapt the Inception architecture to new use-cases.
 
Szegedy et al. described a few general principles and optimization ideas for scaling up the Inception architecture efficiently in (Szegedy, 2016a) :
 
- Higher dimensional representations are easier to process locally within a network. Increasing the activation per tile in  a convolutional network allows for more disentangled features, and thus the network can be trained faster.
- Spatial aggregation can be done over lower dimensional embedding without much or any loss in representation power.
- Avoid representational bottlenecks, especially early in the network.
- Balance the width and depth of the network
 
In light of these guidelines, they proposed Inception V2 and V3 architectures. One outstanding feature of those architectures is the reduction of activation before aggregation.  This feature is implemented due to that the outputs of near-by activation in vision networks are expected to be highly-correlated. And therefore their activation can be reduced before aggregation.  By doing so, similar expressive local representations can still be achieved.
 
The features of Inception V2 and V3 include:
 
- Factorization into smaller convolutions. Similar to VGG Net,  Inception V2 replaces convolutions with large spatial filters  by a two-layered convolutional architecture.  Specifically It replace the 5x5 convolution with two layers of 3x3 convolutions.
 
<div>
<center>
<img  src="/img/2017-05-30/inception_v2_3_01.png">
</center>
<center>Figure 7. Mini-network replacing the 5x5 convolutions. (Szegedy, 2016a) </center>
<br>
</div>
 
<div>
<center>
<img  src="/img/2017-05-30/inception_v2_3_02.png">
</center>
<center>Figure 8.Inception modules where each 5x5 convolution is replaced by two 3x3 convolution. (Szegedy, 2016a) </center>
<br>
</div>
 
-              Spatial factorization into asymmetric convolutions
 
                Convolutions with 2 dimensional filters (e.g., 3x3 filters) can be factorized into asymmetric convolutions.  One example is that, a 3x1 convolution followed by a 1x3 convolution is equivalent to a 3x3 convolution with the same receptive field.  In theory, any n-by-n convolution can be replaced by a 1xn convolution followed by a nx1 convolution. The savings in the computational cost increases as n grows.
 
<div>
<center>
<img  src="/img/2017-05-30/inception_v2_3_03.png">
</center>
<center>Figure 9. Mini-network replacing the 3x3 convolutions. (Szegedy, 2016a)  </center>
<br>
</div>
 
-  Efficient Grid Size Reduction
                Traditionally pooling operations are used to decrease the grid size of the feature maps.  To avoid a representational bottleneck,  the activation dimension of the network filter is expanded.  That means the overall computational cost is dominated on the convolution on the expanded grid.  The solution to reduce the computational cost is to employ two parallel blocks,  pooling and convolutional  layer.
 
<div>
<center>
<img  src="/img/2017-05-30/inception_v2_3_04.png">
</center>
<center>Figure 10. Inception modules that reduces the grid-size while expands the filter banks. (Szegedy, 2016a) </center>
<br>
</div>
 
 
- Model Regularization via Label Smoothing
A mechanism is proposed for encouraging the model to be less confident to alleviate over-fitting.  The idea is to replace the ground-truth label distribution with a mixture of the original ground-truth distribution and a fixed prior distribution on labels.  Such mechanism is termed label-smoothing regularization (LSR).
 
Inception V3, which ships with label smoothing, factorized 7x7, and batch normalization,  reached a top-5 error of 3.585% on the ILSVRC-2012 validation set.
 
 
2016: The Residual networks (ResNet)
==
 
Deep network models have been exploited on ImageNet databases.  However learning better networks is not as easy as stacking more layers.  For one thing, vanishing/exploding gradients hamper convergence from the beginning. Though this problem has been largely addressed by normalized initialization and intermediate normalization layers, accuracy gets saturated with the network depth increasing. Since the degradation of accuracy is not caused by over-fitting, adding more layers leads to higher training error, as depicted in Figure 11.
 
<div>
<center>
<img  src="/img/2017-05-30/res_net_01.png">
</center>
<center>Figure 11. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer plain networks. The deeper network has higher training error, and thus higher test error. (He, 2016)   </center>
<br>
</div>
 
Given a deep learning architecture, its deeper counterpart can be constructed by adding more identity mapping layers on top of its original layers. Due to the identity mapping, such construction solution should produce no higher training error than its shallower counterpart. However experiments show that the current solvers might have difficulties in  approximating identity mappings by multiple nonlinear layers. This prompts the use of residual mapping.
 
The ResNet was proposed (He, 2016) to address the degradation problem by fitting a  residual mapping, instead of the desired underlying mapping.  Formally, Let the desired underlying mapping be H(x),  ResNet let the stacked nonlinear layers fit another mapping F(x) := H(x) -x.  The original mapping is recast into F(x)+x.  It is hypothesized that it is easier to optimize the residual mapping than the original mapping. The formulation of F(x) + x can be realized by "shortcut connections", as shown in Figure 12.
 
<div>
<center>
<img  src="/img/2017-05-30/res_net_02.png">
</center>
<center>Figure 12.  Residual learning: a building block. (He, 2016)  </center>
<br>
</div>
 
 
In ResNet, short cut connections are simply identity mappings, which introduce neither extra parameters nor computational complexity.  As a result, the constructed network can still be trained in an end-to-end manner by SGD with backpropagation.
 
In ResNet, the residual learning is adopted to every few stacked layers.  Note that the form of function is flexible. Experiments in (He, 2016) concern a function F that has two or three layers.  The architecture for ResNet is demonstrated in Figure 13.
 
<div>
<center>
<img  src="/img/2017-05-30/res_net_03.png">
</center>
<center>Figure 13. The ResNet achictecture (He, 2016)  </center>
<br>
</div>
 
The ResNet achieved reached a top-5 error of 3.57% on the ILSVRC-2015 validation set.
 
Summary
==
 
The evolution of CNN dated from 1989, when LeNet was proposed. However due to the insufficient support of hardware for its computational cost,  the development of CNN experienced an incubation time of 20 years.  The introduction of AlexNet in 2012 was a leap in the history of CNN.  AlexNet has inspired its successors, in the sense that some of its features are still shared by them. 
 
A number of CNNS have been developed, based on the architecture of AlexNet, for example, VGG Net (2015),  GoogLeNet (2015),  BN-Inception (2015),  Highway Net(2015) Inception V2 and V3 (2016), and the ResNet(2016). The evolution and relation of these networks are depicted in Figure 14.
 
Generally speaking, they can be divided into two streams. One stream focused on extending the network depth to exploit the expressiveness of depth network; VGG Net, Highway Net, and ResNet are representatives.   Another stream emphasized on developing efficient network architectures by deploying modules of mini-networks inside a network; GoogLeNet, BN-Inception, Inception V2 and V3 all contain  modules of mini-networks. Both streams had achieved gains in performance for ILSVRC.
 
<div>
<center>
<img  src="/img/2017-05-30/cnn_evolution.png">
</center>
<center>Figure 14.   The CNN evolution    </center>
<br>
</div>
 
We have only covered a selected number of CNNs in this blog post. The evolution of CNNs is still on-going. With the advance of GPU, and open-source library, it is expected that the evolution would be happening at a much faster pace then before. 
 
 
References
=
(LeCun, 1989) LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Handwritten digit recognition with a back-propagation network, 1989. In Neural Information Processing Systems. 
 
(Krizhevsky, 2012)  Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems  (pp. 1097-1105).  
 
(Arora, 2014) Arora, S., Bhaskara, A., Ge, R., & Ma, T. (2014). Provable Bounds for Learning Some Deep Representations. In ICML (pp. 584-592). 
 
(Simonyan, 2014)  Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
 
(Szegedy, 2015) Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1-9). 
 
(Loff, 2015) Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015). 
 
(Srivastava, 2015) Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training very deep networks. In Advances in neural information processing systems (pp. 2377-2385) 
 
(Szegedy, 2016)  Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826). 
 
(Szegedy, 2016)  Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2016). Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv preprint arXiv:1602.07261. 
 
(He, 2016) He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778). 
 