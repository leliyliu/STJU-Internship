# SNN

---

## **Towards spike-based machine intelligence with neuromorphic computing**

从神经科学来看，人脑非凡的能力可归结于以下三个基本观察：广泛的连通性、结构和功能化的组织层次、以及时间依赖（time dependent）的神经元突触连接。基于脉冲的时间处理机制使得稀疏而有效的信息在人脑中传递。

人脑的计算原理和硅基计算机之间存在着鲜明区别。其中包括：（1）计算机中计算（处理单元）和存储（存储单元）是分离的，不同于人脑中计算（神经元）和存储（突触）是一体的；（2）受限于二维连接的计算机硬件，人脑中大量存在的三维连通性目前无法在硅基技术上进行模拟；（3）晶体管主要为了构建确定性布尔（数字）电路开关，和人脑基于脉冲的事件驱动型随机计算不同。使得“通用智能”（包括基于云服务器到边缘设备）无法实现的主要瓶颈是巨大的能耗和吞吐量需求。

第三代神经网络主要使用“整合放电”（integrate-and-fire）型尖峰神经元，通过脉冲交换信息。

![image-20200711171632830](https://i.loli.net/2020/07/11/niLocgH8I6fqM7j.png)

第二代神经网络使用了实值计算(real-value)（例如，信号振幅），而SNN则使用信号的时间（脉冲）处理信息。脉冲本质上是二进制事件，它或是0或是1。SNNs中的神经元单元只有在接收或发出尖峰信号时才处于活跃状态，因此它是事件驱动型的，因此可以使其节省能耗。无论实值输入和输出，DLNs所有单位都处于活跃状态。此外，SNN中的输入值为1或0，这也减少了数学上的点积运算ΣiVi×wi，减小了求和的计算量。

SNNs最大的优势在于其能够充分利用基于时空事件的信息。今天，我们有相当成熟的神经形态传感器，来记录环境实时的动态改变。这些动态感官数据可以与SNNs的时间处理能力相结合，以实现超低能耗的计算。在此类传感器中使用SNNs主要受限于缺乏适当的训练算法，从而可以有效地利用尖峰神经元的时间信息。

SNNs的最终能力应当来自于它们处理和感知瞬息万变的现实世界中的连续输入流，就像人脑轻而易举所做的那样。目前，我们既没有良好的基准数据集，也没有评估SNNs实际性能的指标。收集更多适当的基准数据集的研究，例如动态视觉传感器数据或驾驶和导航实例，便显得至关重要。

### SNN 中的学习算法

#### 基于转换的方法

它的基本原理是，使用权重调整（weight rescaling）和归一化方法将训练有素的DLN转换为SNN，将非线性连续输出神经元的特征和尖峰神经元的泄漏时间常数（leak time constants），不应期（refractory period）、膜阈值（membrane threshold）等功能相匹配。

这种方法有其内在的局限性。例如在使用双曲线正切（tanh）或归一化指数函数（softmax）后，非线性神经元的输出值可以得正也可以得负，而脉冲神经元的速率只能是正值。因此，负值总被丢弃，导致转换后的SNNs的精度下降。转换的另一个问题是在不造成严重的性能损失的前提下获得每一层最佳。

#### 基于脉冲的方法

在基于脉冲的方法中，SNN使用时间信息进行训练，因此在整体脉冲动力学中具备明显的稀疏性和高效率优势。大多数依赖反向传播的成果为脉冲神经元功能估计了一个近似可微的函数，从而使其能够执行梯度下降法。

![image-20200711193014782](https://i.loli.net/2020/07/11/zafrZqxkcEGp5by.png)

SpikeProp及其相关变体已派生出通过在输出层固定一个目标脉冲序列来实现SNNs的反向传播规则。

#### 对二进制学习的启示

实际上在算法层级，目前正在研究以概率方式学习（关于神经元何时随机突跳，权重的转换精度何时变低）获得参数较少的网络和计算操作。二元和三元的DLNs也被提出，其神经元输出和权重只取低精度值-1、0和+1，而且在大规模分类任务中表现良好。



### 其它有待研究的方向

#### 终生学习和小样本学习

深度学习模型在长期学习时会出现灾难性遗忘现象。比如，学习过任务A的神经网络在学习任务B时，它会忘记学过的任务A，只记得B。如何在动态的环境中像人一样具备长期学习的能力成为了学术界关注的热点。这固然是深度学习研究的一个新的方向，但我们应该探究给SNN增加额外的时间维度是否有助于实现持续性学习型任务。另一个类似的任务就是，利用少量数据进行学习，这也是SNN能超过深度学习的领域。SNN中的无监督学习可以与提供少量数据的监督学习相结合，只使用一小部分标记的训练数据得到高效的训练结果[ 46,50,65 ]。

#### 与神经科学产生联系

我们可以和神经科学的研究成果相结合，把这些抽象的结果应用到学习规则中，以此提高学习效率。例如，Masquelier等人[65]利用STDP和时间编码模拟视觉神经皮层，他们发现不同的神经元能学习到不同的特征，这一点类似于卷积层学到不同的特征。研究者把树突学习[66]和结构可塑性[67]结合起来，把树突的连接数做为一个超参数，以此为学习提供更多的可能。SNN领域的一项互补研究是LSM（liquid state machines）[68]。LSM利用的是未经训练、随机链接的递归网络框架，该网络对序列识别任务表现卓著[ 69–71]。但是在复杂的大规模任务上的表现能力仍然有待提高。

### 硬件展望

从前文对信息处理能力和脉冲通信的描述中，我们容易假设一套具备类似能力的硬件系统。这套系统能够成为SNN的底层计算框架。

#### “超级大脑”芯片

“超级大脑”芯片[80]的特点是整合了百万计的神经元和突触，神经元和突触提供了脉冲计算的能力[78,81–86]。Neurogrid[82]和TrueNorth[84]分别是基于混合信号模拟电路和数字电路的两种模型芯片。Neurogrid使用数字电路，因为模拟电路容易积累错误，且芯片制造过程中的错误影响也较大。设计神经网络旨在帮助科学家模拟大脑活动，通过复杂的神经元运作机制——比如离子通道的开启和关闭，以及突触特有的生物行为[82,87]。

异步地址事件表示（Asynchronous address event representation）：首先，异步地址事件表示不同于传统的芯片设计，在传统的芯片设计中，所有的计算都按照全局时钟进行，但是因为SNN是稀疏的，仅当脉冲产生时才要进行计算，所以异步事件驱动的计算模式更加适合进行脉冲计算[89,90]。

芯片网络：芯片网络（networks-on-chip，NOCs）可以用于脉冲通信，NOC就是芯片上的路由器网络，通过时分复用技术用总线收发数据包。大规模芯片必须使用NOC，是因为在硅片加工的过程中，连接主要是二维的，在第三个维度灵活程度有限。也要注意到，尽管使用了NOC但芯片的联通程度，仍然不能和大脑中的三维连通性相比。

#### 超越冯·诺依曼计算

晶体管尺寸规模的持续行缩小的现象被称之为摩尔定律[91]，摩尔定律推动了CPU和GPU以及“超级大脑”芯片的不断发展。通过系统总线传输数据。因此，数据在高速的运算单元和低速的存储单元之间的频繁传输就成为了众所周知的“存储墙瓶颈”（memory wall bottleneck）。这一瓶颈限制了计算的吞吐和效率[94]。

减轻这一瓶颈影响的方法就是使用“近内存（near-memory）”、“内存中”计算[95,96]。近内存计算是通过在内存单元附近嵌入一个专门的处理器，由此实现内存和计算的“共存”。实际上，各种“超级大脑芯片”的分布式计算体系结构所具有的紧密放置的神经元和突触阵列就是近内存计算的表现。相比较而言，内存中计算则是把部分计算操作嵌入到内存内部或外部电路中。

#### 非易失性技术

非易失性技术（ non-volatile technology）[97–103]通常被用于与生物突触相比较。实际上，它们展示了生物突触的两个特征：突触效能（synaptic efficacy）和突触可塑性（ synaptic plasticity）。突触可塑性指的是根据特定的学习规则调整突触权重的能力。突触效能指的是根据输入脉冲产生输出的现象。以最简单的形式来说，意思就是，输入的脉冲信号乘以突触的权重。这表示着可编程、模拟、非易失性。从上游神经元得到的信号，相乘再求和后再作用于下游神经元的输入。

虽然原位计算（situ computing）和突触学习为大规模超越冯·诺依曼分布式计算提供了诱人的前景，但有许多的挑战仍然有待克服，因设备、因周期和进程相关引起的变化，计算的近似性质容易出现错误，从而减低整体的计算效率，最终影响准确性。此外，交叉开关操作的鲁棒性受到电流潜通路、线电阻、驱动电路的源电阻和感测电阻的存在的影响[117,118]。选择器（晶体管或双端非线性装置）的非理想性、对模拟-数字转换设备的要求和有限的比特精度要求，也增加了使用非传统突触装置设计可靠计算的总体复杂性。

### 算法-硬件协同设计

#### 混合信号模拟计算

模拟计算容易受到过程引起的变化和噪声的影响，并且由于模拟和数字转换设备的复杂性和精度要求，在面积和能耗方面就受到了很大限制。将芯片学习与紧密结合的模拟计算框架相结合，将使这类系统能够从根本上适应过程引起的变化，从而减轻对精度的影响。

更好的容错局部学习算法——即使是要学习额外的参数——将是推动模拟神经形态计算的关键所在。

#### 混合设计方法

我们认为，基于混合方法的硬件解决方案——即在单一平台上结合各种技术的优势——是另一个需要深入研究的重要领域。这种方法可以在最近的文献中找到[137]，比如，把低精度忆阻器与高精度数字处理器结合使用。这种混合方法有许多可能的变体，包括显著驱动的计算数据分离、混合精度计算[137]、将常规硅存储器重新配置为需内存近似加速器[125]、局部同步和全局异步设计[138]、局部模拟和全局数字系统；其中新兴技术和传统技术可以同时使用，以提高精确度和效率。此外，这种混合硬件可以与基于混合脉冲学习的方法结合使用，例如局部无监督学习，然后是全局有监督反向传播算法[53]。我们认为，这种局部-全局学习方案可以用来降低硬件复杂性，同时，最大限度的减少对终端应用程序的性能影响。
