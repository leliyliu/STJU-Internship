[TOC]

# GE 后端执行Example

---

在所给出的GE的源码中，有给出一个`ResNet50`执行的例子，其具体位置在`tests/st/resnet50`中，下面对其进行相应的分析。 

其相应的python 代码实际是直接到了`resnet50_train.cc`中，然后进行相应的执行，首先看其`main`函数，具体为：

```c++
int main(int argc, char *argv[]) {
  // add loop for test of stabilty:
  int loopCount = 1;
  if (argc >= 2) loopCount = atoi(argv[1]);

  Status ret = SUCCESS;
  ret = runTrainGraph(resnet50, loopCount);
  if (ret == SUCCESS) {
    std::cout << "[train resnet50 success]" << std::endl;
  } else {
    std::cout << "!!! train resnet50 fail !!!" << std::endl;
  }
  return ret;
}
```

故而实际上到了GE 这一层次，实际上ME端将相应的图传到了GE端，然后执行，故而要执行的第一步即为`runTrainGraph`。 可以查看其调用逻辑，具体可查看[runTrainGraph 调用逻辑](architecture/runTrainGraph.vdx)

