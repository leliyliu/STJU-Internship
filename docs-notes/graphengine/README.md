[TOC]

# GraphEngine(GE) in MindSpore

---

GE 是整个MindSpore 框架中的一个子模块，主要是用来连接前端和设备端，主要是用C++ 来开发实现的。 GE模块从前端（ME）得到相应的图，然后将其作为输入，并将一系列图操作拿出来适应一个特定的形式，使其能够在设备上高效地执行。GE实际上是专门设计来作为一个对于Ascend 板卡上的有效操作的部件， 它是被自动调用的，不会暴露在用户面前。 其主要包括了两个部分，分别是GE API 和 GE core 。 下面是关于GE 的一个简单架构描述：

![GE_Architecture](D:\leliy\github_project\STJU-Internship\docs-notes\graphengine\GE_Architecture.png)

### GE API

GE API 是CE core 和前端之间的一个接口，用于控制GE core的初始化和结束的相应操作。 同时其提供了图添加和图运行的相应接口。 

### GE CORE

GE core 是整个GE 的一个主模块， 负责整个图的相应处理操作。 其主要包括了六个部分， 有graph preparation, graph partition, graph optimization, graph compilation, graph loading 和 graph execution。这六个部分是线性执行的，并且综合在一起来完成整个复杂的图处理的操作。  

#### 1. Graph preparation

在图中的所有的特征图和变量的shape 都在这一阶段被推导出来， 并为后续的内存分配做准备。 一些像allreduce 之类的聚集操作也都在这里进行执行（执行图的分发准备）。 整个的Ascend 芯片是一个异构芯片，包括了CPU ， 向量计算单元， AI core 等。 在图中的每一个操作都会根据相应硬件的支持和消耗被分配到一个固定的执行核中。

#### 2. Graph partition

整个的大图会被分成许多个子图，基于在之前一个阶段所分配的操作。会加上一些相应的标记来记录子图的边界，这样的一个子图使得可以有一个有效的优化。 

#### 3. Graph optimization

根据不同子图所属的engines，会采用不同的优化接口来进行优化。 为了完整地利用整个AI core 中的CUBE 的计算能力（具体请参看Ascend 芯片设计与 Da Vinci 架构）， 更改了数据布局和其相应的传输，使得其能够更好的适应数据的读取操作。 这种操作保证了在RAM 和 CUBE 之间的更好的数据处理。除此之外，一些算子也进行了相应的融合，使之成为了一个单个的大的操作，来减少计算开销。 

#### 4. Graph compilation

这一阶段实际上可以被分成两个部分，包括资源分配和图的编译。 在内存分配中，充分考虑了内存重用的相应机制。 根据图的相应信息，队列，事件，流资源被相应进行了分配。 每个相应的任务都会被编译指定到一个特定的流当中。 在同一个流当中的相应任务会按照顺序线性执行， 在不同流当中的任务会被并行执行。 流的分配会在这个阶段完成。

#### 5. Graph loading

根据engine 信息， 图中的操作被分配到不同的engine， 然后图被加载到device 中来运行。

#### 6. Graph execution 

在这个阶段， 图中的操作在device 中执行，并且相应的输出会返回到主机端。为了性能考虑， 一个整图下沉的模式被提出，是在执行了多次之后才返回相应的输出结果。 这样的一个模式能够减少数据传输之间的开销。