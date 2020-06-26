[TOC]

# 第二讲 GraphEngine 解析

---

嵌入在前端和晟腾芯片之间的一个重要模块。 

## 深度学习芯片简介

关于scalar, vector , cube 的相应理解，对于一个相应的矩阵乘法操作，可以看到，其具体应该如下所示：

```c++
float a[16][16],b[16][16],c[16][16];
for(int i=0;i<16;i++)
    for(int j=0;j<16;j++)
        for(int k=0;k<16;k++)
            c[i][j] += a[i][k] * b[k][j];
            
for(int i=0;i<16;i++)
    for(int j=0;i<16;j++)
        c[i][j] = a[i][:] *= b[:][j];

c[:][:] = a[:][:] X b[:][:]
```

![image-20200626132853976](https://i.loli.net/2020/06/26/em79qnOz3TPdHhB.png)

## GraphEngine

![image-20200626133353297](https://i.loli.net/2020/06/26/i9r1KaAWGYhUEjB.png)



#### On-Device Execution(整图下沉)

将整张的计算图全部放到芯片上去执行，减少芯片和host之间进行交互的次数。 

![image-20200626133602315](https://i.loli.net/2020/06/26/lChZ8HBm4Q5vRLn.png)

host CPU 将整个图一次性发送给晟腾芯片，注意对于整个晟腾芯片而言，其中有一个CPU core 来进行相应的控制，因此在整个模型计算完成之前，其不会将数据返回到host CPU端。  

![image-20200626133836041](https://i.loli.net/2020/06/26/R6xBKat45vYgkUr.png)

可以看到，由于晟腾芯片上有很大的内容，并且保持了很宽的数据通路，其设计了相应的HCCS协议，并支持其它一些协议方式。整体而言，GE的优化是一个软硬件协同优化。 

循环下层，即多个step来返回数据，而非一个step 来进行返回。 

#### Pipeline Parallel（并行计算）

单纯的并行策略， 并行的读取数据，和部分分布式并行进行反向计算。 其最基本的思想就是通过简单的并行计算，来隐藏掉部分耗时的计算。

![image-20200626140830763](https://i.loli.net/2020/06/26/xAiW4pMREU8QKZd.png)

#### Low level Optimization（深度图优化）

图在GE的整个流程上承载了所有信息，任何网络执行需要的网络都可以从不同阶段的图中找到。 

GE图的基本元素：Node、Anchor、Attributes、Graph

其中这个图没有显式的边的概念，而是通过Anchor来进行相应的映射，得到一个相应的映射关系。

GE图的格式：

1. prototxt格式： 方便搜索、查看、对比算子
2. Onnx格式： 可使用Netron等软件打开，方便详细分析较小的子图

##### 图管理

对于整张图而言，其具体的操作可以包括：

![image-20200626141551832](https://i.loli.net/2020/06/26/ift23rsAjRCkHFw.png)

将一整张大图中的内容进行相应的划分，将算子映射到相应的计算引擎（硬件）， 然后根据相应的内容进行子图拆分，并插入边界标识。 格式转换与算子融合的工作，采用5D 的格式，使得在AI core中的搬运更加有效，算子的融合也能减少数据搬运时间。 每个task 要绑定在相应的流上面， 将图加载到硬件上，调用runtime接口返回结果。 

##### 动态图与静态图（Graph Mode 和 Pynative Mode）

context.set_context(mode = context.GRAPH_MODE) 或者是 context.PYNATIVE_MODE

 静态图显然能够更快进行硬件上的执行，而动态图会更加灵活。 

## Debug工具

GE提供数据dump和图dump的功能，以方便调试，还包括profiling的相应工具。



TBE工具用来自定义算子