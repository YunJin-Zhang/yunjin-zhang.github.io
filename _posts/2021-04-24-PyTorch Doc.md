---
title: PyTorch Doc
layout: post
categories: pytorch
tags: pytorch
date: 2021-04-24 23:00
excerpt: PyTorch Doc
---

# PyTorch Doc

{:.table-of-content}
* TOC
{:toc}

## Torch

对tensor进行判断运算、设置默认类型。

创建tensor、创建随机多维tensor、0值tensor等。

对tensor进行处理，如索引、切片、聚合、降维、reshape等。

创建随机种子、随机状态、概率分布等、特定尺寸的多维tensor等。

`torch.Tensor`中还内嵌了更多分布形式。



存储对象和加载对象（由`torch.save()`保存的对象）。

并行进程控制。

允许/禁止梯度的局部线程控制。



对tensor的数学运算、逻辑运算、比较运算、排序操作、在频域上的操作（如哈明窗）。

以及传播、拷贝、**对于矩阵的操作**。



对于单个矩阵和两个矩阵的运算、计算线性方程组的解、PCA降维等等

## Torch.nn

(Uninitialized)Parameter

模型、模块、参数列表的容器，以及提供模块的全局hook。

Flatten层。

CNN层。

pooling层。

padding层。

activation函数层。

normalization层。

RNN层。

Transformer层。

线性函数层。

Dropout层。

embedding层。

相关性计算层（如余弦距离）。

损失函数层。

重排列、撤销重排列、上采样层。

数据并行层。

`torch.nn.utils`：对于梯度的操作、参数向量化、权重标准化、频谱标准化、prune操作

`torch.utils.rnn`：对于**序列**的pad、pack操作。

## Torch.nn.functional

卷积functions：conv1d、conv2d、conv3d；conv_transpose1d、conv_transpose2d、conv_transpose3d（反卷积）；unfold & fold。

池化functions：avg/max/lp/adaptive max/adaptive avg 1d/2d/3d。

非线性activation functions：threshold、relu、hardtanh、hardwish、softmax...

标准化functions

线性funcitons

dropout functions

稀疏functions：embedding、embedding bag、one-hot。

距离functions：pairewise、cosine...

损失functions：entropy、smooth l1...

视觉functions：pixel shuffle/unshuffle、pad、interpolate、unsample、grid、affine...

数据并行functions：

## Torch.tensor

tensor分类：dtype、CPU tensor、GPU tensor

常用：`torch.Tensor.item()`、`torch.device`、

创建Tensor：`torch.tensor()`、`torch.*`、`torch.\*_like`、`tensor.new__\*`。

方法：判断属性、查看grad等、对tensor做运算、对tensor做操作、对tensor作比较、基于index的操作、对tensor的类型转换、对tensor做判断、masked操作、筛选操作、requries_grad、dui tensor进行函数运算、迁移操作、ubind/unfold...

## Tensor Attributes

`torch.dtype`、`torch.device`、`torch.layout`(stride)、`torch.memory_format`

## Tensor Views

一个已存在的tensor的 `View` 版本避免显式的数据复制，可进行快速reshaping、切片、element-wise等操作。

view操作 in PyTorch:基本切片和索引、 `as_strided()`、`detach()`、`diagnal()`、`squeeze()`、`contiguous()`...

## Torch.autograd

只要声明 `Tensor` 的 `requires_grad=True` 关键字即可自动计算梯度. 只支持浮点数和复数`Tensor` 的自动梯度。

`torch.autograd.backward`

`torch.autograd.grad`

**torch.autograd.functional**：`torch.autograd.functional.jacobian/hessian/vjp/jvp/vhp/hvp`

**局部禁止梯度计算**：`torch.autograd.no_grad/enable_grad/set_grad_enabled`

**默认梯度layouts**：当一个non-sparse参数在反向传播中收到non-sparse梯度时，参数梯度的积累方式。也可**手动梯度layouts**。

**Tensor上的In-place操作**

**Tensor autograd functions**：`grad\requires_grad\is_leaf\backward\detach\register_hook\retain_grad\forward`

**Context method mixins**：

**数值上的梯度检查**

**profiler**

**异常检查**

## Torch.cuda

支持对于利用GPU的计算，支持CUDAtensor类型。使用`is_available()`来决定自己的系统是否支持CUDA。

**对于CUDA配置的操作**

**随机数字生成器**

**通信collectives**：数据在多个GPU上的通信配置

**流和事件**

**内存管理**

**NVIDIA工具拓展**

## Torch.cuda.amp

对于混合精度操作的配置。

## Torch.backends

对于pytorch支持的不同后端的行为的控制。

## Torch.distributed

分布式并行计算的配置。 [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) 

## Torch.distributions

包含了可参数化的分布和采样函数。可容许随机计算图的设立和对于优化的随机梯度估算器。

**打分函数**：![image-20210419201350703](C:\Users\张赟瑾\AppData\Roaming\Typora\typora-user-images\image-20210419201350703.png)

```python
probs = policy_network(state)
# Note that this is equivalent to what used to be called multinomial
m = Categorical(probs)
action = m.sample()
next_state, reward = env.step(action)
loss = -m.log_prob(action) * reward
loss.backward()
```

**pathwise微分**：

The other way to implement these stochastic/policy gradients would be to use the reparameterization trick from the `rsample()` method, where the parameterized random variable can be constructed via a parameterized deterministic function of a parameter-free random variable. The reparameterized sample therefore becomes differentiable. The code for implementing the pathwise derivative would be as follows:

```python
params = policy_network(state)
m = Normal(*params)
# Any distribution with .has_rsample == True could work based on the application
action = m.rsample()
next_state, reward = env.step(action)  # Assuming that reward is differentiable
loss = -reward
loss.backward()
```

**Distribution**

**ExponentiaFamily**

**Bernoulli**

**Beta**

**Binomial**

**Categorical**

**Cauchy**

**KL-Divergence**

...

**Transforms**

**Constraints**

**Constraint Registry**：有两个全局Constraint Registry来连接Constraint对象与Transform对象。

## Torch.fft

## Torch.futures

## Torch.fx

## Torch.hub

是一个预训练好的模型集。

 [pytorch/vision repo](https://github.com/pytorch/vision/blob/master/hubconf.py)

**从Hub中加载模型**：`torch.hub.list/help/load/download_url_to_file/load_state_dict_from_url/`、

## Torch.jit

是从Pytorch代码中创建可序列化的、可优化的模型的方法。任何一个TorchScript程序都可从一个没有Python依赖的过程中被保存并加载。

## Torch.linalg

线性代数运算

Cholesky分解；condition数；方阵的行列式；方针的行列式；方针的行列式的和符号和自然对数；复矩阵的特征值和特征向量；矩阵的秩；矩阵的norm；矩阵的伪逆；矩阵的SVD分解；矩阵方程的解；tensor的input_inv；矩阵的逆矩阵；矩阵的QR分解；

## Torch.overrides

展示了对于不同__torch_function__的协议的帮助函数。

## Torch.profiler

可查看训练过程和推理过程中的性能度量。可用于更好的理解那些模型运算符是成本更高的，查看它们的输入形状和stack轨迹，设备kernel活跃度，可视化执行轨迹。

## Torch.nn.init

`torch.nn.init.calculate_gain`：返回给定非线性函数的推荐增益值；

`torch.nn.init.uniform_/normal_/constant_/ones_/eye_dirac_/xavier_uniform_/xavier_norm_/kaiming_uniform_/kaiming_normal_/orthogonal_/sparse_/`：用均匀/正态/分布/常数中的值填充进输入tensor；

## Torch.onnx

在不同框架上的模型。

## Torch.optim

包含不同优化算法。如要使用.cuda()，则应在建构优化器之前就将模型迁移至cuda。

**优化算法**：`torch.optim.Optimizer/Adadelta/Adagrad/Adam/AdamW/SparseAdam/Adamx/ASGD/LBFGS/RMSprp/Rprop/SGD/`。方法：`add_param_group()`、`load_state_dict()`、`state_dict()`、`step()`、`zero_grad`。

**调整步长**：`torch.optim.lr_scheduler`提供了几种调整步长的方法。应在优化器更新后再应用learning rate scheduling。`torch.optim.lr_scheduler.LambdaLR/MultiplicativeLR/StepLR/MultiStepLR/ExponentialLR/CosineAnnealingLRReduceLROnPlateau/CyclicLR/OneCycleLR/CosineAnnealingWarmRestarts`。方法：`load_state_dict`、`state_dict`。

```python
scheduler = ...
for epoch in range(100):
	train(...)
	validate(...)
	scheduler.step()
    
# example
>>> # Assuming optimizer has two groups.
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```

**随机权重平均SWA**： In particular, `torch.optim.swa_utils.AveragedModel` class implements SWA models, `torch.optim.swa_utils.SWALR` implements the SWA learning rate scheduler and `torch.optim.swa_utils.update_bn()` is a utility function used to update SWA batch normalization statistics at the end of training.

## Complex Numers

## DDP Comunication Hooks

DDP Communication Hooks是一种通过在[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.)中override vanilla allreduce来在所有workers上传播梯度的通用interface。有些hooks是built-in的，也可通过定制来优化通信。

## Pipeline Parallelism 

Pipeline parallelism是一种在多GPU上训练大型模型的高效技术。

## Quantization

与模型压缩表示和高性能向量化操作相关。

- Torch.nn.intrinsic：combined modules conv+relu which can be quantized.

- Torch.nn.intrinsic.qat：versions of those fused operations for quantization aware training.

- Torch.nn.intrinsic.quantized：quantized implementations of fused operations like conv+relu.
- Torch.nn.qat：versions of the key nn modules **Conv2d()** and **Linear()** which run in FP32 but with rounding applied to simulate the effect of INT8 quantization.
- Torch.quantization：implements the functions you call directly to convert your model from FP32 to quantized form. 
- Torch.nn.quantized： implements the quantized versions of the nn modules and functionals.
- Torch.nn.quantized.dynamic：linear，LSTM，LSTMCell，GRUCell，RNNCell。

## Distributed RPC Framework

通过一系列primitives提供对于多机器模型训练的体制，以实现远距离通信，并提供一个更高级的API来自动区分不同机器上的模型。

## Remote Reference Protocol

描述了远距离参考协议的设计细节，并且走过了在不同场景中的信息流。在阅读本篇之前请保证你已熟悉 [Distributed RPC Framework](https://pytorch.org/docs/stable/rpc.html#distributed-rpc-framework) 。

## Torch.random

`torch.random.fork_rng`：当返回时，RNG被重设为之前在的那个状态。

`torch.random.get_rng_state/initial_seed/manual_seed/seed/set_rng_state`：返回随机数生成器状态/生成随机数的初始种子/设置随机数的种子/设置随机数（非deterministic随机数）的种子/设置随机数生成器状态。

## Torch.sparse

默认的设置中，数组在内存中都是邻接存储的，然而，有一种很重要的多维属组叫做稀疏属组，如果用邻接内存存储这些数组元素的话，得到的效率和效果都不如人意。稀疏数组中很多元素都是零值，因此如果只存储非零值的话，内存就会很大程度上被节省。有许多稀疏存储格式（COO，CSR/CSC，LIL等等）。

## Torch.storage

`torch.Storage`是一种单个数据类型的邻接的、一维数组。

## Torch.utils.benchmark

`torch.utils.benchmark.Timer`：测量Pytorch程序的执行时间的类。

`torch.utils.benchmark.Measurement`：存储了给定描述的更多测量方法。是Timer测量的结果。

`torch.utils.benchmark.CallgrindStats`：由Timer收集的Callgrind结果的最高级容器。

`torch.utils.benchmark.FunctionCounts`：控制Callgrind结果的容器。

## Torch.utils.bottleneck

可被用作在程序中debug瓶颈的初始步骤。

用以下命令行debug：

```python
python -m torch.utils.bottleneck /path/to/source/script.py [args]
```

## Torch.utils.checkpoint

![image-20210420120238578](C:\Users\张赟瑾\AppData\Roaming\Typora\typora-user-images\image-20210420120238578.png)

`torch.utils.checkpoint.checkpoint`：为模型添加检查点。添加了检查点的部分不保存中间activation，反而是在反向计算中重新计算它们。在前向计算中，函数根据`torch.no_grad()`来运行，也就是不存储中间activation。

`torch.utils.checkpoint.``checkpoint_sequential`：对于检查点sequential模型的帮助函数。

## Torch.utils.cpp_extension可

对于C++的拓展。

## Torch.utils.data

核心是Pytorch的数据加载工具[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。可用于：map方式和迭代方式的数据集；定制数据加载命令；自动batching；单线程或者多线程数据加载；自动内存pinning。

[`torch.utils.data.Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)用于规定数据加载中的索引/keys序列。

`torch.utils.data.Dataset`：表示数据集的类。用于创建子类。

`torch.utils.data.IterableDataset`：用于创建可迭代的数据集的子类。

`torch.utils.data.TensorDataset/ConcatDataset/ChainDataset/BufferedShuffleDataset/Subset/`

`torch.utils.data.get_worker_info`：返回当前Dateloader迭代器工作线程的信息。

`torch.utils.data.random_split`：切割数据集。

`torch.utils.data.Sampler/SequentialSampler/RandomSampler/SubsetRandomSampler/WeightedRandomSampler/BatchSampler/DistributedSampler`

## Torch.utils.dlpack

## Torch.utils.mobile_optimizer

`torch.mobile_optimizer.optimize_for_mobile`可通过eval模式中的模块来执行一系列优化器pass。包括以下参数： 一个torch.jit.ScriptModule 对象，一个优化器的黑名单，以及一个 被保留的method列表。它也会唤起只保留了前向方法的freeze_moduel pass。

## Torch.utils.model_zoo

移至了torch.hub。

## Torch.utils.tensorboard

[more details]( https://www.tensorflow.org/tensorboard/)

安装Tensorboard后，可在TensorBoard中进行可视化。如Scalars、images、histograms、graphs和嵌入可视化。

安装命令：

```python
pip install tensorboard
tensorboard --logdir=runs
```

`torch.utils.tensorboard.writer.SummaryWriter`

## Type Info

`torch.finfo`：是一个展示浮点数数值特性的对象。

`torch.iinfo`：是一个展示整数的数值特性的对象。

## Named Tensors

可指定tensor维度，可自动检查API的是否正确使用。也可用于重设置维度名，以支持通过名字的传播机制而不是通过位置的传播机制。还可重排列维度名位置等。

## Named Tensors Operator Coverage

对于被命名的tensor的参考姓名的操作说明：使用姓名以提供额外的自动runtime正确性检查；从输入tensor开始传播姓名至输出tensor。

## TorchVision

`torchvision`包含了流行的数据集、模型结构和CV的常用图像转换。

- `torchvision.datasets`：包含了流行的数据集。

- `torchvision.io`：用于读取或写入视频和图像。`torchvision.io.read_video/wirte_video/VideoReader/`；`torchvision.io.decode_image/encode_image/encode_jpeg/write_ipeg/encode_png/wirte_png`

- `torchvision.models`：包含了流行的对于图像分类的模型。

- `torchvision.ops`：包含了对于计算机视觉的运算符：NMS，IOU，Align等。

- `torchvision.transforms`：Transforms就是普通意义上的图像转换。可与 [`Compose`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose)结合使用。[`torchvision.transforms.functional`](https://pytorch.org/vision/stable/transforms.html#module-torchvision.transforms.functional)模块则在所有的转换上都具有fine-grained控制。这在当你建立了一个更加复杂的转换Pipeline时十分有用。所有转换都接受PIL图像作为输入。Tensor图像是形如$$(C, H, W)$$的Tensor，具有形如$$(B, C, H, W)$$的Tensor图像集合中的$$B$$意为一批次中图像的数量。在批次中的Tensor图像上应用的deterministic或随机变换会将批次内所有图像转换。Scriptable transformers（Sequential），Compositions of transforms（not torchscript），对于PIL图像和 Torch Tensor上的的转换，Conversion/Generic/Functional Transforms。

- `torchvision.utils`：`torchvision.utils.make_grid/save_image/draw_bounding_boxes/`

