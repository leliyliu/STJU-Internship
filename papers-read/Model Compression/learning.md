# 神经网络压缩与ReRAM

---

#### attention 机制

关于attention 机制以及其在CNN中的相应应用，建议参考此文章[深度学习中的注意力机制](https://zhuanlan.zhihu.com/p/53036028)，其中着重强调了attention机制的发源以及其在CNN中如何应用此机制以达到好的结果

#### 神经网络压缩

对于神经网络压缩，有众多的方法，在这里我们主要研究的是量化剪枝以及其在硬件上的应用， 其具体方法这里不多加考虑。 对于神经网络压缩的一个简要方法综述，可以参考[深度学习模型压缩与加速综述](https://zhuanlan.zhihu.com/p/67871864)，实际上，对于现有的深度模型压缩方法，主要可以分为四类：

- 参数修剪和共享（parameter pruning and sharing）
- 低秩因子分解（low-rank factorization）
- 转移/紧凑卷积滤波器（transferred/compact convolutional filters）
- 知识蒸馏（knowledge distillation）

## 量化

### DEEP COMMPRESSION : Compressing deep neural networks with pruning, trained quantization and huffman coding

这篇文章主要讲述了进行网络压缩的三个步骤，首先进行剪枝，然后进行量化，最后通过霍夫曼编码。 

#### Introduction

深度神经网络已经在大量的任务上取得了目前最好的结果，然而其的权重过大，并且有大量的能量消耗，因此，其很难部署在移动设备当中。而实际上，很大程度上的能量消耗来自于内存的开销。为了减少存储和能量开销，运行推理网络，提出了一个三阶段的深度压缩的概念（deep compression)。

![image-20200623220712718](https://i.loli.net/2020/06/23/7Qgs6oJGj8nykiC.png)

#### network pruning

网络剪枝对于减小网络的复杂性和避免过拟合有着很好的效果。 对于剪枝而言，其主要分为这样几步：

+ 通过普通的网络训练来学习整个的连接性
+ 减掉那些小权重的连接（低于一个threshold的都会被减掉）
+ 重新训练整个网络

对于一个稀疏的权值矩阵而言，其中大多数值为0， 因此采用相应的存储稀疏矩阵的方式进行存储，注意理解这里存储的方法，其大小为$2a + n + 1$，故而其实际上存储方式为（以CSC为例）：

对于一个$n\times n$的矩阵而言，其中有$a$个值非0，那么对于存储而言，首先需要存储$a$个非0值，然后是其每个值对应的相应列$a$，对于每一行而言，其从第几个非稀疏值开始，共$n+1$，所以为此，具体可参考[Deep compression论文讲解](https://blog.csdn.net/weixin_36474809/article/details/80643784)。 在压缩完之后，存储权值和权值对应的参数，然而其保存的不是绝对参数，而是相对参数，即两个参数之间的差值，只要两个元素小于8，那么可以存为3 bit 的，而如果大于，就在8的位置处置0。 

#### Trained quantization and weight sharing 

网络量化和权重共享进一步压缩了剪枝后的网络，通过减少需要的bit 数。通过共享权重， 我们限制了有效的需要存储的权重，并且有效调整了这些权重。 

如图所示，假设拥有一个4 input neurons 和 4 output neurons， 那么权重就是$4 \times 4$的矩阵，在该图左上是一个$4 \times 4$ 的权重矩阵，而左下则是一个梯度矩阵，这个权重呗量化到4种不同的bins 中（不同颜色标识），那么只需要存储的是一个2-bit 的index 矩阵，在更新过程中，梯度会根据颜色来相加得到共享的中心点。 对于剪枝后的AlexNet，对于每一卷积层，我们能够量化一个8-bits的矩阵(256 个共享权重），而全连接层，就能到5-bits(32个共享权重 -> idx)

![image-20200623230346190](https://i.loli.net/2020/06/23/ahRV1Ic3xy45Usd.png)

为了计算整个的压缩率， 对于给定的k 个簇，只需要$\log_2(k)$的bit 位来编码整个的index，总体而言，对于一个拥有$n$个connections，并且每个connections用b bits来表示的网络， 压缩率可以表示为：
$$
r = \frac{nb}{n\log_2(k) + kb}
$$
对于图中所给出的例子而言，实际上$n=16,k=4,b=32$，根据计算，可以得到其压缩率为3.2倍

##### WEIGHT SHARING

利用K-means聚类来标识这些共享权值，因而所有属于同一个cluster的权重将共享权值。在层与层之间，权重是不共享的。 而对于所有一个簇的权重，其用中心值的权重来进行代替。

##### INITIALIZATION OF SHARED WEIGHTS

中心点的初始化影响到了整个聚类的效果，因而测试了三种不同的初始化方法： Forgy(random),density-based, 和 linear initialization。 对于权值的分布进行测试，可以看到，对于AlexNet 的conv3 层，在剪枝之后其呈现一个双峰分布。

![image-20200624080503524](https://i.loli.net/2020/06/24/o2eEIbWRHN7f8dj.png) 

+ frogy : 随机进行初始化点，由于在两峰分布密集，因此随机值也更倾向于在两峰之中。 
+ Density-based: 在y轴上进行线性选择，然后找到其对应的CDF 中的相应x轴坐标点。这种方法使得点依然密集于两峰之间，然而分布的更零散
+ Linear： 在x轴进行线性的选择（根据最大值和最小值，以及K 值），分布过于零散，没有体现峰值。 

根据实验表明，线性初始化考虑到了那些最大的权重，故而在实验中是最优的。

##### FEED-FORWARD AND BACK-PROPAGATION

一个关于共享权值的表的index 被保存在连接之间， 在反向传播阶段，对于每一个共享权重的梯度都会被计算并且用来更新共享权重。假设$L$代表loss，应该有：
$$
\frac{\partial L}{\partial C_k} = \sum_{i,j}\frac{\partial L}{\partial W_{ij}}\frac{\partial W_{ij}}{\partial C_k} = \sum_{i,j}\frac{\partial L}{\partial W_{ij}}f(I_{ij} = k)
$$

#### Huffman coding

利用霍夫曼编码来减少内存，大约能减少20% - 30%



除此之外，该论文还讨论了关于剪枝，量化单独进行与同时进行的准确率损失效果。 并展示了不同的中心点初始化能给出的结果。 这里还进行了一系列相关的讨论，不再一一说明，下面给出结论

#### Conclusion

+ 简单的方法也可以达到很好的效果
+ 这种方法能够使得有很大的可能让神经网络运行在移动端

## 剪枝

### Learning Structured Sparsity in Deep Neural Networks

这篇文章主要提出了一个结构化稀疏学习的方法（SSL），其主要包括以下优点：

+ 将一个大的DNN网络压缩到一个紧致的网络中以减小计算损失
+ 一个硬件友好的结构化稀疏，而能够有效进行加速（实验中，在CPU和GPU中平均能够有5.1 和 3.1倍的加速， AlexNet，利用现成库)，这比非结构化的稀疏要快两倍
+ 正则化了DNN的结构以提高分类准确率

#### Introduction

传统的稀疏化正则化和连接剪枝，通常导致非结构化的剪枝结果，因此导致内存读取影响到了实际的加速过程。 受到如下的影响：

+ 在filters 和 channels 中有冗余
+ 传统的filters被设定为长方体，然而一个随意形状的能够有效减少不必要的计算
+ 网络深度是必要的，但是更深的网络不能总保证一个更好的error rate，因为梯度消失和爆炸的缘故

#### Structured Sparsity Learning Method for DNNs

在这里，主要关注于SSL 在卷积层上的优化， 来正则化DNN结构。

##### Proposed structured sparsity learning for generic structures

假设卷积层的权重为： $W^{(l)} \in \R^{N_l \times C_l \times M_l \times K_l}$， 这里的$l$代表的是DNN 的层数。那么所提出的一般优化目标为：
$$
E(W) = E_D(W) + \lambda \cdot R(W) + \lambda_g \cdot \sum_{l=1}^LR_g(W^{(l)})
$$
在这里，W代表了所有DNN网络中的权重的集合， $E_D(W)$代表了数据的损失， $R(\cdot)$是非结构化正则化应用在每层上（例如l2 正则化） ， 而$R_g(\cdot)$则是每一层的结构化正则化方法。其中，根据正则化方法， 有$R_g{(w)} = \sum_{g=1}^G ||w^{(g)}||_g$，其中$w^{(g)}$是$w$中的部分权重并且G是group的数量。 不同组之间可能会有重合。

##### Structured sparsity learning for structures of filters, channels, filter shapes and depth

在SSL中，学习到结构化实际上是利用对于group的划分得到的。通过移除掉一些group，就能够得到一个更紧致的DNN。

+ 移除不重要的filters 和 channels ，可根据论文中的公式找到其中对于$E(W)$影响不大的group
+ 学习任意形状的filters:  通过不同的n来进行判断，而得到随意形状的filter
+ 约减网络层数： 这和前两种的稀疏化方法是不同的，因为输出为0会导致feature map 无法工作，一次你提出一种权衡网络之间的shortcuts来解决这个问题

##### Structured sparsity learning for computationally efficient structures

除了上述提出的方法来完成一个紧致的压缩之外，上述公式的变体同样也能有一些相应的效果。

3D卷积网络实际上是2D卷积网络拼接而成的， 为了得到高效的卷积，探索了一个稀疏度的细粒度的变体，被称为2D-filter -wise 稀疏。 

#### Conclusion

提出的这一种SSL的方法能够正则化 filter, channel, filter shape 和DNN 中的层深度，这种方法能够显著加速DNN网络，不论是在CPU还是在GPU上，用其原有的库。除此之外， SSL的一个变体能够被用来提升准确率。 

## 低bit训练

## ReRAM



