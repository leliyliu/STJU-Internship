[TOC]

# 第一讲： MindSpore分布式自动并行训练

### 数据并行与模型并行

#### 数据并行

![image-20200625163744881](https://i.loli.net/2020/06/25/meOb7IRkyC9DxA6.png)

将模型在不同板卡上进行相应的复制，对于batch 进行分布。 实现训练间的同步，要进行梯度聚合，要注意数据同步。 

![image-20200625165134755](https://i.loli.net/2020/06/25/6Bd7N1Q5UyCzHxG.png)

![image-20200625165149735](https://i.loli.net/2020/06/25/5lvXJHQ6Bxk2iaS.png)

#### 模型并行

模型内存过大，分为层内模型并行和层间模型并行，对于mindspore 而言，其主要支持层内模型并行的方式。 相比于数据并行而言，模型并行的难度更大，对开发者的要求更高。考虑内存上限，在性能上要兼顾通信开销，还要关心张量排布，切分的维度。 主流的框架主要通过用户自主切分的方式，以pytorch为例，其要通过手动实现相应的代码，具体如下：

![image-20200625165747145](https://i.loli.net/2020/06/25/IRqliLcCWO7y9kK.png)

### MindSpore自动并行方案

![image-20200626122031273](https://i.loli.net/2020/06/26/ol6BMRwantkWS8e.png)

对于MindSpore自动并行方案而言，其框架主要如上图所示， 其打破了样本和参数的便捷，按计算数据维度进行相应的切分，实现混合并行。

#### 算子自动切分

算子自动切分即通过相应的对于算子切分策略的搜索，得到较好的切分策略， 并在算子之间，通过相应的`Tensor redistribution`来实现相应的集合通信

在MindSpore中提供了相应切分策略的代码，其接口为`set_strategy()`，具体实现为：

```python
set_strategy([d0,d1,d2],[d2,d1,d0]) # 第一个是input 切分， 第二个是weight 切分
# 对于数据并行而言
set_strategy([dev_num,1,1],[1,1,1])
# 对于模型并行而言
set_strategy([1,1,1],[1,1,dev_num])
```

具体的一个模型并行转数据并行

![image-20200626124528614](https://i.loli.net/2020/06/26/KoRBcw58WgT4jXx.png)

其它的例子可以参考白皮书。 

#### 基于策略搜索的自动切分

整个MindSpore而言，其基于Ascend910 的相应架构，建立模型并提出了一个高校的并行策略搜索算法，具体如图所示：

![image-20200626124851980](https://i.loli.net/2020/06/26/at4UedomisczNXu.png)

#### 分布式自动微分

在利用分布式框架进行计算时，其利用了自动微分流程，实现分布式反向的自动生成，避免复杂的手动微分操作。 即不仅在单机中有自动微分，在分布式框架中也可以采用自动微分的方式来进行训练。

![image-20200626125057351](https://i.loli.net/2020/06/26/g7LGSCVZmyeoYqw.png)

![image-20200626125123228](https://i.loli.net/2020/06/26/2sZhMRbfXoBALiF.png)

对于整个的auto Parallel ，其接口的调用如下所示：

```python
context.set_auto_parallel_context(parallel_mode = ParallelMode.AUTO_PARALLEL)
```

![image-20200626131433697](https://i.loli.net/2020/06/26/o91ipqfwCRxb3J2.png)