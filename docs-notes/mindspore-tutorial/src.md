# 第三讲 MindSpore代码流程分析

---

![image-20200626152841925](https://i.loli.net/2020/06/26/l4eWqk1nFXUoRCt.png)

#### 样例代码

![image-20200626153319429](https://i.loli.net/2020/06/26/nWPc8t6CRB1xfQF.png)

样例代码目前主要支持八个模型，可以看到其对应的硬件支持和分布式训练支持等相应内容。

![image-20200626153332786](https://i.loli.net/2020/06/26/qRsgOyP3TVEKbDC.png)

#### Python 组件

![image-20200626153533844](https://i.loli.net/2020/06/26/uFAQSw1c3BRPvLm.png)

其中context，记录了系统运行中的环境变量，包括GPU，自动并行等相应操作。 

train 和 nn 目录下的相应定义，主要是基于训练中的相应算子和相应模型和export方法用作模型导出。 

ops 目录中定义了一些常用的算子， 其执行的方式的不同

communication : 主要是与分布式并行有关的数据通量等。 

#### C++ 组件

![image-20200626154202426](https://i.loli.net/2020/06/26/fLuPi2YbmDQ8VdO.png)