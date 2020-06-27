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