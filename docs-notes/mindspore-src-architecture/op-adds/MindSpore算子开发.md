[TOC]

# MindSpore 算子开发

---

对于MindSpore而言，其提供了多种算子开发策略，因此要分多种不同情况进行相应的解析。 

## 基于CPU 的简单MindSpore算子开发

在具体开发之前，首先简单介绍一下MindSpore当前的算子以及其源码的相应结构。 关于MindSpore 的安装，可以参考[MindSpore安装方法](https://www.mindspore.cn/install/)，目前而言，MindSpore支持还不是很全面，而且开源内容也不完整，作为简单的测试，基于每个人都能上手的硬件设备，就从Win10 下的CPU 安装（因为GPU 目前只支持 Ubuntu 版本，后续在服务器上再进行相应考虑）。 首先介绍一下在当前环境下的相应限制： 针对于CPU的环境，目前无法支持动态图，只能使用静态图进行相应的操作，因此在进行相应算子测试的时候，只能采用静态图方式来进行测试，这在后面我们将更加清晰地认识到。 除此之外，可以看到，当前的MindSpore 框架对于算子支持的力度，实际上主要实现了在Ascned 上对于算子的支持，而对于GPU 和CPU ，其支持还较少，尤其是针对于CPU而言，其具体可以参考[MindSpore算子支持](https://www.mindspore.cn/docs/zh-CN/master/operator_list.html)。 在进行具体的CPU  MindSpore算子开发之前，可以首先通过一个算子测试过程来简单了解一下算子运行时的基本流程，其具体可参考MindSpore源码架构中的相应内容。 



