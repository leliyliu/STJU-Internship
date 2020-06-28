[TOC]

# MindSpore 源码架构

---

关于MindSpore的源码结构，我们将首先通过一些较为简单的例子来进行相应的分析和理解，而后再整体来重新构建整个MindSpore源码相应的分析，以及一些相应的补充部分。 在具体开始了解MindSpore源码之前，首先请查看[网络定义约束](https://gitee.com/mindspore/docs/blob/master/docs/source_zh_cn/constraints_on_network_construction.md)。

## One-Hot 测试

### 测试代码

在进行代码的分析之前，首先查看在`test\st\ops\cpu\test_one_hot_op.py`中的相应代码，具体如下所示：

```python
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pytest
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.api import ms_function
import numpy as np
import mindspore.context as context

context.set_context(device_target='CPU')


class NetOneHot(nn.Cell):
    def __init__(self):
        super(NetOneHot, self).__init__()
        self.on_value = 2.0
        self.off_value = 3.0

        self.depth_1 = 6
        self.one_hot_1 = nn.OneHot(-1, self.depth_1, self.on_value, self.off_value)

        self.depth_2 = 4
        self.one_hot_2 = nn.OneHot(0, self.depth_1, self.on_value, self.off_value)
        self.one_hot_3 = nn.OneHot(0, self.depth_2, self.on_value, self.off_value)
        self.one_hot_4 = nn.OneHot(1, self.depth_1, self.on_value, self.off_value)

    @ms_function
    def construct(self, indices1, indices2, indices3, indices4):
        return (self.one_hot_1(indices1), self.one_hot_2(indices2),
                self.one_hot_3(indices3), self.one_hot_4(indices4))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_one_hot():
    one_hot = NetOneHot()
    indices1 = Tensor(np.array([[0, 1], [4, 5], [2, 6]]),mindspore.int32)
    indices2 = Tensor(np.array([1, 2, 3]),mindspore.int32)
    indices3 = Tensor(np.array([[0, 1], [1, 0]]),mindspore.int32)
    indices4 = Tensor(np.array([[0, 1], [4, 5], [2, 6]]),mindspore.int32)
    output = one_hot(indices1, indices2, indices3, indices4)
    expect_0 = np.array([
        [[2., 3., 3., 3., 3., 3.], [3., 2., 3., 3., 3., 3.]],
        [[3., 3., 3., 3., 2., 3.], [3., 3., 3., 3., 3., 2.]],
        [[3., 3., 2., 3., 3., 3.], [3., 3., 3., 3., 3., 3.]]
    ]).astype(np.float32)
    expect_1 = np.array([
        [3., 3., 3.],
        [2., 3., 3.],
        [3., 2., 3.],
        [3., 3., 2.],
        [3., 3., 3.],
        [3., 3., 3.]
    ]).astype(np.float32)
    expect_2 = np.array([
        [[2., 3.], [3., 2.]], [[3., 2.], [2., 3.]], [[3., 3.], [3., 3.]],
        [[3., 3.], [3., 3.]]
    ]).astype(np.float32)
    expect_3 = np.array([
        [[2., 3.], [3., 2.], [3., 3.], [3., 3.], [3., 3.], [3., 3.]],
        [[3., 3.], [3., 3.], [3., 3.], [3., 3.], [2., 3.], [3., 2.]],
        [[3., 3.], [3., 3.], [2., 3.], [3., 3.], [3., 3.], [3., 3.]]
    ]).astype(np.float32)
    assert (output[0].asnumpy() == expect_0).all()
    assert (output[1].asnumpy() == expect_1).all()
    assert (output[2].asnumpy() == expect_2).all()
    assert (output[3].asnumpy() == expect_3).all()
    print(output[3].asnumpy().shape)

if __name__ == '__main__':
    test_one_hot()
```

可以看到，利用`context`设置了相应执行设备为CPU，而非GPU 或者 Ascend，这块的相应内容在后面还可以看到，这里暂时先不用太在意`context`中的相应内容。 可以看到，虽然没有直接声明其为静态图，但是利用类的定义，实际上是一个静态图，如果采用[MindSpore算子支持](https://www.mindspore.cn/docs/zh-CN/master/operator_list.html)中的相应代码，是无法完成测试的，因为该定义方式为动态图的定义方式，这里一定要注意。 

### nn 接口代码

这里实际上调用了`nn.OneHot`的相应接口，可以看到源码为：

```python
class OneHot(Cell):
    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype=mstype.float32):
        super(OneHot, self).__init__()
        self.onehot = P.OneHot(axis)
        self.depth = depth
        self.on_value = Tensor(on_value, dtype)
        self.off_value = Tensor(off_value, dtype)

    def construct(self, indices):
        return self.onehot(indices, self.depth, self.on_value, self.off_value)
```

对于所有类，都要首先继承Cell，这是一定要记得的。下面主要对于这些参数进行相应的说明：

```
indices : 需要进行one-hot 编码的原内容
axis: 从哪个维度开始进行one-hot 编码，default = -1 
depth: 深度，即用几个数来进行one-hot 编码
on_value: default = 1， 即数字对应的相应位置
off_value: default = 0 , 即非数字对应的相应位置，与on_value 相反
```

### operations 以及 基类分析

在这里实际上调用了`P.OneHot`函数，那么继续查相应的内容： 

```python
class OneHot(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, axis=-1):
        self.init_prim_io_names(inputs=['indices', 'depth', 'on_value', 'off_value'], outputs=['output'])
        validator.check_type("axis", axis, [int])

    def __infer__(self, indices, depth, on_value, off_value):
        # check type
        validator.check_subclass("indices", indices['dtype'], mstype.tensor)
        validator.check_typename("indices", indices['dtype'], (mstype.int32,))
        validator.check_typename("depth", depth['dtype'], mstype.int_type)
        validator.check_subclass("on_value", on_value['dtype'], mstype.tensor)
        validator.check_subclass("off_value", off_value['dtype'], mstype.tensor)
        args = {"on_value dtype": on_value['dtype'], "off_value dtype": off_value['dtype']}
        validator.check_type_same(args, (mstype.float16, mstype.float32))

        # check shape
        indices_shp = indices['shape']
        validator.check_int_range("axis", self.axis, -1, len(indices_shp), Rel.INC_BOTH)
        depth_val = depth['value']
        validator.check_integer("depth", depth_val, 0, Rel.GE)
        # create new dimension at end if self.axis is -1
        indices_shp.insert(self.axis, depth_val) if self.axis >= 0 else indices_shp.append(depth_val)

        return {'shape': indices_shp,
                'dtype': on_value['dtype'],
                'value': None}
```

上面的类进行了几个相应的操作，包括通过调用初始化函数进行初始化，然后使用`validator`这一实例进行了输入类型的验证（其中`validator`是类`ParamValidator`的实例，该类主要的内容就是进行参数类型的验证，其中的函数就是对各类参数进行验证，具体内容可以参考`mindspore\_checkparam.py`文件），在`__init__`中只是验证了`axis`参数，而在`__infer__`参数中对于`indices`等进行了相应的验证，可以参考上述代码。  

通过`MindSpore`网络约束的相关文件，我们可以知道，实际上`mindspore/ops/operations/*`中的相关代码，都是`Primitive`算子，故都需要继承`PrimitiveWithInfer`或者`Primitive`类本身。

#### Primitive

`mindspore\ccsrc\ir\primitive.cc` 这里面实现了相应的内容，是一个对应的接口。`mindspore\ccsrc\dataset\api\python_binding.cc`是算子进行绑定的相应部分内容 。

```python
def __init__():
    name, attrs , init_attrs  # 相应的name , 属性和初始化属性
def _fill_signature(self,signatures): # 首先填相应的signature
def __call__(self.*args):
    output = _run_op(self,self.name,args)
    # 这是其中最重要的一个函数，实际上就是调用了_run_op函数，然后执行相应的算子操作，对于其他的部分，就暂时不作分析了。

    
# 调用_run_op，实际上是调用了real_run_op 函数，其具体内容应该在类PrimitiveWithInfer当中，下面具体看一下这个代码
@_wrap_func #注意使用了wrap_func
def _run_op(obj, op_name, args):
    """Single op execution function supported by ge in PyNative mode."""
    op_mask = [0] * len(args)
    op_inputs = []
    for i, arg in enumerate(args):
        if hasattr(arg, '__parameter__'):
            op_inputs.append(arg.default_input)
            op_mask[i] = 1
        else:
            op_inputs.append(arg)
    output = real_run_op(obj, op_name, tuple(op_inputs), tuple(op_mask))
    if not output:
        raise RuntimeError("Pynative run op %s failed!" % op_name)
    if len(output) == 1:
        output = output[0]
    return output
```

可以看到，在这里将相应的obj，op操作和相应的input 进行了传入，并调用了相应的`real_run_op`函数，这里实际上就从python 代码转到了 c 代码，下面开始进行相应的分析。

可以看到其具体继承的相应位置：

```python
from .._c_expression import Primitive_, real_run_op, prim_type
from .._c_expression import signature_rw as sig_rw
from .._c_expression import signature_kind as sig_kind
from .._c_expression import signature_dtype as sig_dtype

# 并且其通过了相应的strategy 来设定了算子计算的方式
```

### 算子注册

对于MindSpore新实现相应算子，应该对其进行注册，一般而言，当前需要实现的算子主要是MindSpore框架下还不支持的，考虑其是否有在上述的代码中进行相应的声明，如果已经进行了相应的声明，那么实际上需要考虑的内容只是关于算子在CPUKernel（这里以CPU算子为例，如果想要开发GPU算子，可以参考[MindSpore GPU 算子开发](https://www.bilibili.com/video/BV1cV411d72A)） 的注册和其具体逻辑的实现。

#### one_hot_cpu_kernel.h

正如上面所说的，需要首先对于算子进行相应的注册，在MindSpore中，定义了相应的注册算子的宏，为:`MS_REG_CPU_KERNEL()`，其中包含了两个参数，第一个参数为注册的算子的名称，第二个为具体实现的相应Kernel，具体代码可以如下所示：

```c
#ifndef MINDSPORE_CCSRC_DEVICE_CPU_ONE_HOT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_DEVICE_CPU_ONE_HOT_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "device/cpu/cpu_kernel.h"
#include "device/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace device {
namespace cpu {
class OneHotCPUKernel : public CPUKernel {
 public:
  OneHotCPUKernel() = default;
  ~OneHotCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  size_t depth_;
  size_t stride_;
  size_t axis_;
};

MS_REG_CPU_KERNEL(OneHot, OneHotCPUKernel);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_CPU_ONE_HOT_CPU_KERNEL_H_
```

可以看到，主要需要包含四个函数，分别为构造函数，析构函数，初始化Kernel函数与Launch函数，并且其继承自`CPUKernel`，主要的计算内容在`Launch`函数当中，下面主要查看一下实现该函数的代码

#### one_hot_cpu_kernel.cc

```c
bool OneHotCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 3 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "input or output invalid!";
  }
  auto indices = reinterpret_cast<int *>(inputs[0]->addr);
  auto on_value = reinterpret_cast<float *>(inputs[1]->addr)[0];
  auto off_value = reinterpret_cast<float *>(inputs[2]->addr)[0];
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(int);

  for (size_t i = 0; i < elem_num; i++) {
    size_t stride_num = i / stride_;
    size_t output_index = stride_num * depth_ * stride_ + i % stride_;
    size_t index = IntToSize(indices[i]);
    for (size_t j = 0; j < depth_; j++) {
      if (index == j) {
        output[output_index] = on_value;
      } else {
        output[output_index] = off_value;
      }
      output_index += stride_;
    }
  }
```

关于输入的参数类型，其为`AddressPtr`，其中包含了两个参数，addr 指向该数据的地址，也就是该数据的相应指针，而size就是指该数据规模 。 可以看到主要包括了三个输入参数，分别为`indice`,`on_value` 和 `off_value`，其实现逻辑很简单，即调用了`InitToSize`，来获得相应的index，并判断其是否在depth范围之类，如果是指向该index，那么值设置为`on_value`，否则为`off_value`。 

至此，看到了在MindSpore中一个CPU算子是如何开发的，并且了解到了简单的调用接口，那么借此来认识MindSpore框架。 

## MindSpore 框架分析

对于其中相应模块先进行简要说明：

```
_extends : 相应的一些扩展
akg : 
ccsrc : C 代码的相应实现，是ME中偏底层的一些模块
common : 一些公用模块
communication : 一些通信相应的模块
dataset : 相应数据集
mindrecord : 进行相应的记录的内容
model_zoo : 
nn : mindspore nn 模块的相应算子 python 接口
ops : mindspore operations python 接口
parallel : 进行相应并行化控制的模块
train : 进行训练的模块
```

对于其架构的理解，也可以从此图中得到一些相应的认识：

![MindSpore-architecture](https://i.loli.net/2020/06/28/AjHcTMUV91Klmqp.png)

可以看到，实际上上述的代码主要是与ME层进行相应的对应的

### Python 组件

#### mindspore/context.py

```python
# context.py
__all__ = ['GRAPH_MODE', 'PYNATIVE_MODE', 'set_context', 'get_context', 'set_auto_parallel_context',
           'get_auto_parallel_context', 'reset_auto_parallel_context']
# GRAPH_MODE 静态图 PYNATIVE_MODE 动态图
# set_context 设置相应context 内容  
# get_context 查看当前的context 中的相应内容
# set_auto_parallel_context 设置一个自动并行的环境
# get_auto_parallel_context 获得当前自动并行环境的内容
# reset_auto_parallel_context 重置环境
```

##### 补 ： 静态图与动态图

实际上静态图和动态图的概念，可以与两种不同的编程方式相联系来进行分析，其分别为：

```
动态图 ：命令式编程
静态图 ：符号式编程
```

下面给出使用python 代码进行命令式编程和符号式编程的相应例子：

**命令式编程**

```python
import numpy as np
a = np.ones(10)
b = np.ones(10) * 2
c = b * a
d = c + 1
```

从这里可以看到，在每一步中，都会执行相应的操作，比如进行赋值和相乘和相加等，即执行到相应代码的时候，就会做对应的数据计算。

对于命令式编程而言，其显然更加灵活，其也和我们平常编写python 代码的方式十分相似。

**符号式编程**

```python
A = Variable('A')
B = Variable('B')
C = B * A
D = C + Constant(1)
# compiles the function
f = compile(D)
d = f(A=np.ones(10), B=np.ones(10)*2)
```

可以看到，对于计算而言，其只需要在`d = f(A=np.ones(10), B=np.ones(10)*2)`这一部分才需要具体执行，而不是在`C = B * A`这一步就会计算出相应的C 的结果，而显然，这是与上述所给出的命令式编程的方式不同的。

而相对于命令式编程而言，符号式编程的方式则更加高效，其与计算机原生语言更加贴合。

在基于上面的了解后，可以简单认识到实际上动态图能够动态的执行，即图还没有完全构建完就进行了相应的执行， 而且也很容易从中间导出相应结果，编程框架非常灵活； 而相对而言，静态图则需要从计算一开始就知道整个图的架构（或者说整个图保持静态），所以底层能对其进行一些有效的优化。 

关于静态图和动态图更多相应区别的理解，可以参考[深度学习框架中的静态图和动态图](https://blog.csdn.net/qq_36653505/article/details/87875279)。对于`MindSpore`而言，其支持动态图和静态图的相应转换，在整图下沉部分采用静态图，而也可以采用动态图编程模式进行代码编写。目前来说，实际上可以采用Python 中的(JIT) 编译器来编译命令式程序，获得一些符号式编程对全局信息掌握的优势。这也正是`MindSpore`中采用的动态图与静态图转换的方式。

##### set_context

在代码中，给出了一些相应进行设置的例子，如下所示：

```python
    Examples:
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> context.set_context(mode=context.PYNATIVE_MODE)
        >>> context.set_context(device_target="Ascend")
        >>> context.set_context(device_id=0)
        >>> context.set_context(save_graphs=True, save_graphs_path="./model.ms")
        >>> context.set_context(enable_task_sink=True)
        >>> context.set_context(enable_mem_reuse=True)
        >>> context.set_context(enable_reduce_precision=True)
        >>> context.set_context(save_ms_model=True, save_ms_model_path=".")
        >>> context.set_context(enable_gpu_summary=False)
        >>> context.set_context(enable_dump=False, save_dump_path=".")
        >>> context.set_context(reserve_class_name_in_scope=True)
        >>> context.set_context(enable_dynamic_memory=True)
        >>> context.set_context(graph_memory_max_size="25GB")
        >>> context.set_context(variable_memory_max_size="6GB")
        >>> context.set_context(mode=context.GRAPH_MODE,
        >>>                     device_target="Ascend",device_id=0, save_graphs=True,
        >>>                     save_graphs_path="/mindspore")
```

即，可以设置图的模式，包括静态图和动态图， device 包括了 Ascend ， CPU ，GPU 等， 还可以设置内存重用，任务下沉，图保存等，具体可以参考源码内容。 

#### train & nn

以一个具体实例来进行相应的说明：

```python
import mindspore.nn as nn
from mindspore.trainn import Model,export

net = Net()
loss = nn.SoftmaxCrossEntropWithLogits()
optim = nn.Momentum(paramms = net.trainable_params(), learning_rate = 0.1, momentum = 0.9)
model = Model(net,loss_fnn = loss , optimizer = optim , metrics = None)
model.train(2,get_train_dataset())
model.eval(get_eval_dataset()) #model evaluate 
model.predict(get_input_data()) # model inference
export(net,get_innput_data(),file_name = 'net.onnx',file_format = ONNX)
```

可以看到，实际上train 部分主要是用来做训练模型和模型导出部分，而nn就是相应的算子，包括loss 的算子等。 

#### ops 

关于op 主要包括了function 和 operations 两个部分，而实际上对于这一块的调用主要是用来做相应的验证内容。这里主要是针对于模型的算子做一些python 相应的接口，主要需要注意的是，对于动态图和静态图两种图，有不同的写法，在这里一定要进行注意。 这在前面也进行过相应的交代。 

#### common

这一部分是一些低阶的api，包括tensor  的定义等相关内容，进行查看可以很清楚的看到。 

#### communication

主要是和分布式并行相关的api，主要是进行通信等，比如ALLReduce 等需要调用相应的通信机制来完成，就需要这一模块中的相应内容。 

### C++ 组件

![image-20200628171205248](https://i.loli.net/2020/06/28/iOXrZ6qcVMvFdA9.png)

### 代码调用流程

针对于不同的过程，进行其代码调用流程的相应分析，如下所示：

#### CPU模型训练

![image-20200628171440652](https://i.loli.net/2020/06/28/ZBEuLsCGr8ORWKn.png)

可以看到，通过调用model 的相应训练的内容，根据相应的数据下沉模式和图模式，选择其进行处理的方法

##### train -> train_dataset_sink_process

首先实际上对于model 的训练，调用了如下代码：

```python
# /train/model.py
def train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True):
    repeat_count = train_dataset.get_repeat_count()
    if epoch != repeat_count:
		# ...
        if context.get_context("device_target") in ["CPU", "GPU"] and context.get_context("enable_loop_sink"):
            raise ValueError("CPU and GPU can't support loop sink, please set enable_loop_sink=False.")

            self._train(epoch,
                        train_dataset,
                        callbacks=callbacks,
                        dataset_sink_mode=dataset_sink_mode)
```

可以看到，其进行了相应的检查之后调用`self._train`模块进行了后续训练内容，具体如下所示：

```python
# /train/model.py
    def _train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True):
# ...
        if dataset_sink_mode and context.get_context("mode") == context.GRAPH_MODE:
            self._train_dataset_sink_process(epoch, train_dataset, list_callback, cb_params)
        else:
            self._train_process(epoch, train_dataset, list_callback, cb_params)
```

故实际上这里就是图中所显示的相应判断内容，下面以`_train_dataset_sink_process`为主路径进行相应分析

##### train_dataset_sink_process

```python
# /train/model.py
def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None):
    dataset_helper = DatasetHelper(train_dataset)
    # remove later to deal with loop sink
    if need_wrap:
        # ...
        cb_params.train_network = self._train_network
        self._train_network.set_train()

        cb_params.cur_step_num = 0
        loop_size = dataset_helper.loop_size()
        run_context = RunContext(cb_params)
        _callback_wrapper(list_callback, run_context, "begin")

        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1
            _callback_wrapper(list_callback, run_context, "epoch_begin")

            # for data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                cb_params.cur_step_num += loop_size
                _callback_wrapper(list_callback, run_context, "step_begin")
                outputs = self._train_network(*inputs)
                cb_params.net_outputs = outputs
                _callback_wrapper(list_callback, run_context, "step_end")

                _callback_wrapper(list_callback, run_context, "epoch_end")
                should_stop = should_stop or run_context.get_stop_requested()
                if should_stop:
                    break

                    _callback_wrapper(list_callback, run_context, "end")
```

可以看到，在这里，其首先调用了`datasetHelper`函数来进行相应的数据初始化，而在相应执行过程中，每次都会回调`RunContext()`来进行整体环境的执行，而对于每一次的输入数据，则会执行`_train_network`来得到相应的结果。 下面来对两个主要部分分别进行分析

##### DatasetHelper

其实际代码内容如下：

```python
# /model/dataset_helper.py
def __init__(self, dataset, dataset_sink_mode=True):
    check_bool(dataset_sink_mode)

    iterclass = _DatasetIterGE
    if not dataset_sink_mode:
        iterclass = _DatasetIterFeed
        elif not context.get_context("enable_ge"):
            if context.get_context("enable_loop_sink"):
                iterclass = _DatasetIterMSLoopSink
                else:
                    iterclass = _DatasetIterMS

                    self.iter = iterclass(dataset)
```

可以看到，在相应类的初始化函数中，其构造了一个`iterclass(dataset)`类来处理相应的内容。 而在该类的构造函数中，其调用了`_exec_datagraph(dataset,self.loop_size)`函数来进行相应的处理。 具体代码如下：

```python
dataset.__ME_INITED__ = _exec_datagraph(dataset, self.loop_size).queue_name
# /train/_utils.py
def _exec_datagraph(exec_dataset, dataset_size, phase='dataset'):
    _executor.init_dataset(exec_dataset.queue_name,
                           dataset_size,
                           batch_size,
                           dataset_types,
                           dataset_shapes,
                           input_indexs,
                           phase=phase)
```

故而，这里实际上调用的是`common`模块中的_executor 来进行初始化数据集的相应工作，继续查看相应代码：

```python
def init_dataset(self, queue_name, dataset_size, batch_size, dataset_types, dataset_shapes,
                     input_indexs, phase='dataset'):
        if not init_exec_dataset(queue_name=queue_name,
                                 size=dataset_size,
                                 batch_size=batch_size,
                                 types=dataset_types,
                                 shapes=dataset_shapes,
                                 input_indexs=input_indexs,
                                 phase=phase):
            raise RuntimeError("Failure to init and dataset subgraph!")
        return True
```

这里继续调用相应的内容，实际上从Python接口转到了C++ 接口，即转到c++的`pipeline::InitExecDataset()`函数当中。其具体代码为：

```c++
// ccsrc/pipeline/pipeline.cc
bool InitExecDataset(const std::string& queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr>& types, const std::vector<std::vector<int64_t>>& shapes,
                     const std::vector<int64_t>& input_indexes, const std::string& phase) {
  std::string name = MsContext::GetInstance()->backend_policy();
  if (name == kMsConvert || name == kMsVm) {
    return InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes);
  } else {
    return InitExecDatasetGe(queue_name, iter_num, batch_size, types, shapes, input_indexes, phase);
  }
}
//然后通过ccsrc/transform中的相关内容来进行具体的初始化数据的操作
```

![image-20200628185135780](https://i.loli.net/2020/06/28/iOG3TzI9PCokefq.png)

可以看到起具体调用过程为：

```c++
bool InitExecDatasetVm(const std::string& queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr>& types, const std::vector<std::vector<int64_t>>& shapes,
                       const std::vector<int64_t>& input_indexes) {
  MS_LOG(INFO) << "Start InitDataSet Entry";
  std::vector<int> int_input_indexes;
  (void)std::transform(input_indexes.begin(), input_indexes.end(), std::back_inserter(int_input_indexes),
                       [](int64_t item) { return static_cast<int>(item); });
  std::vector<std::vector<int>> int_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(int_shapes),
                       [](const std::vector<int64_t>& item) { std::vector<int> vector_item;
                         (void)std::transform(item.begin(), item.end(), std::back_inserter(vector_item), [](int64_t inner_item) { return static_cast<int>(inner_item); });
return vector_item; });

  const std::vector<std::string> emply_str_list;
  p_init->set_attr("input_names", MakeValue(emply_str_list));
  p_init->set_attr("output_names", MakeValue(emply_str_list));

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  auto app_init = std::make_shared<CNode>(AnfNodePtrList{NewValueNode(p_init)}, func_graph);
  func_graph->set_output(app_init);
  auto manager = MakeManager();
  manager->AddFuncGraph(func_graph);

  // AbstractNone indicates there is no output for this apply node.
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  app_init->set_abstract(abstract_none);

  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  auto convert_fn = backend->convert_fn();
  MS_EXCEPTION_IF_NULL(convert_fn);
  // Convert CNodeList to LinConvertResult.
  ConfigManager::GetInstance().set_iter_num(1);
  auto runner = convert_fn({app_init});
  backend->Link(runner.graph_id);
  ConfigManager::GetInstance().set_iter_num(size);

  if (!(*runner.run)) {
    // empty function
    MS_LOG(EXCEPTION) << "Backend " << backend->name() << " unsupports tdt dataset.";
  }

  // launch init dataset runner without inputs and outputs
  VectorRef args;
  auto fn = runner.run;
  (void)(*fn)(args);
  MS_LOG(DEBUG) << "InitDataSetVm End.";
  return true;
}
```

可以看到，调用了三个相应的函数，分别进行相应分析：首先调用了`FuncGraphManager`来进行图的添加， 然后调用`compile::CreateBackend()`来进行了相应后端的处理，这里就不再继续过多介绍相应内容。 

##### _train_network

实际上的_train_network  为 _bulid_train_network 这一具体的函数，代码为：

```python
self._train_network = self._build_train_network()
# /train/model.py
def _build_train_network(self):
    """Build train network"""
    network = self._network
    if self._optimizer:
        if self._loss_scale_manager_set:
            network = amp.build_train_network(network,
                                              self._optimizer,self._loss_fn,level=self._amp_level,
                loss_scale_manager=self._loss_scale_manager)
            else:
                network = amp.build_train_network(network,   self._optimizer, self._loss_fn, level=self._amp_level)
 	elif self._loss_fn: network = WithLossCell(network, self._loss_fn)
  return network
```

故而实际上，这里的内容是调用了`amp.bulid_trainn_network`，而对于其相应的执行，最终需要调用到`nn.Cell`的`__call__`函数中，其具体为：

```python
def __call__(self, *inputs):
    if context.get_context("mode") == context.GRAPH_MODE:
        out = self.compile_and_run(*inputs)
        return out
    return self.construct(*inputs)
```

这个接口会根据图的不同，调用不同的方式，显然对于CPU而言，其只能使用静态图，所以会调用`compile_and_run`函数， 得到的相应结果为：

```python
def compile_and_run(self, *inputs):
    _, compile_flag = _executor.compile(self, *inputs, phase=self.phase)

    if _get_parallel_mode() in ["auto_parallel", "semi_auto_parallel"]:
        if inputs and isinstance(inputs[0], Tensor) and inputs[0].virtual_flag and (not compile_flag):
            parallel_inputs_run = self._parallel_inputs_run
            else:
                self._parallel_inputs_run = self._load_inputs(*inputs)
                parallel_inputs_run = self._parallel_inputs_run
                return _executor(self, *parallel_inputs_run, phase=self.phase)
            return _executor(self, *inputs, phase=self.phase)
```

故实际上，可以看到的是， 如果是自动并行或者半自动并行方式，其会首先进行并行内容的设置，即调用相应的接口来进行实现，否则就直接调用`_executor`来执行模型训练输入内容，即首先会执行`_executor.compile()`来进行相应操作，最后实际上跳转到`/common/api.py`当中，然后从Python接口转到c++接口来进行相应的处理。 其实际上是在`pipeline`的代码当中，可以查看到，其具体为：

```c++
bool ExecutorPy::Compile(const py::object& obj, const py::tuple& args, const py::object& phase, bool use_vm) {
  bool ret_value = false;

  try {
    MS_LOG(DEBUG) << PrintArgs(args);
    ret_value = CompileInner(obj, args, phase, use_vm);
  } catch (const py::error_already_set& ex) {
    // print function call stack info before release
    std::ostringstream oss;
    trace::TraceGraphInfer();
    trace::GetInferStackInfo(oss);
    // call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    ReleaseResource(phase);

    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const std::exception& ex) {
    ReleaseResource(phase);
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    ReleaseResource(phase);
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << exName;
  }

  return ret_value;
}
```

![image-20200628202526349](https://i.loli.net/2020/06/28/UTVSfCGgscEqBMm.png)

![image-20200628202915097](https://i.loli.net/2020/06/28/FzcbyBTfOZAuwvP.png)

![image-20200628203025989](https://i.loli.net/2020/06/28/qd12OThD6W3cg5s.png)

![image-20200628203233992](https://i.loli.net/2020/06/28/QpBi1XY2tDoq54f.png)

#### GPU 单算子执行

![image-20200628203405615](https://i.loli.net/2020/06/28/KUMpizyvLgu4fBZ.png)

#### 导出ONNX 模型

![image-20200628203504405](https://i.loli.net/2020/06/28/ifqRdbc84UX5T62.png)

 通过这些相应的调用逻辑，可以基本了解MindSpore的源码框架，对于其更多的实现逻辑，还要涉及到GE端，所以在接下来进行相应讨论。