---
title: The Transformer Family
layout: post
categories: deep-learning
tags: deep-learning transformer attention
date: 2021-04-15 08:16
excerpt: The Transformer Family
---

# The Transformer Family
本文翻译自lilianweng的博客[The Transformer Family](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html)

> 近期，研究者们在Transformer模型的不同增强版本上取得了很大进展。出于此，本篇博客展现了vanilla Transformer是如何在更长期的attention span、更少的内存和计算量消耗、RL任务的解决上有所提升的。

从我上一篇关于 [attention](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)的博客算起，大概有两年时间了。最近在Transformer的新版本和增强版本的发展让我对于这个主题有了写另一篇博客的欲望，主要是关注vanilla Transformer是如何在更长期的attention span、更少的内存和计算量消耗、RL任务的解决上等面向上有所提升的。

{:.table-of-content}
* TOC
{:toc}

## Notations

| Symbol                                                       | Meaning                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $$d$$                                                          | 模型尺寸/隐状态维度/位置编码尺寸                             |
| $$h$$                                                          | 多头attention层中的头数量                                    |
| $$L$$                                                          | 输入序列被分割成的长度                                       |
| $$\mathbf{X} \in \mathbb{R}^{L \times d}$$                     | 输入序列，其中每个元素都被映射成为一个形状为$$d$$的嵌入向量    |
| $$\mathbf{W}^k \in \mathbb{R}^{d \times d_k}$$                 | key权重矩阵                                                  |
| $$\mathbf{W}^q \in \mathbb{R}^{d \times d_k}$$                 | query权重矩阵                                                |
| $$\mathbf{W}^v \in \mathbb{R}^{d \times d_v}$$                 | value权重矩阵。我们总有$$d_k=d_v=d$$。                         |
| $$\mathbf{W}^k_i, \mathbf{W}^q_i \in \mathbb{R}^{d \times d_k/h}; \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$$ | 单个头上的权重矩阵                                           |
| $$\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$$                 | 输出权重矩阵                                                 |
| $$\mathbf{Q} = \mathbf{X}\mathbf{W}^q \in \mathbb{R}^{L \times d_k}$$ | query嵌入输入                                                |
| $$\mathbf{K} = \mathbf{X}\mathbf{W}^k \in \mathbb{R}^{L \times d_k}$$ | key嵌入输入                                                  |
| $$\mathbf{V} = \mathbf{X}\mathbf{W}^v\in \mathbb{R}^{L \times d_v}$$ | value嵌入输入                                                |
| $$S_i$$                                                        | 与第$$i$$-th个query $$\mathbf{q}_i$$互相关的key位置的集合        |
| $$\mathbf{A} \in \mathbb{R}^{L \times L}$$                     | 长度为$$L$$的输入序列和它自身之间的self-attention矩阵。$$\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$$。 |
| $$a_{ij} \in \mathbf{A}$$                                      | query $$\mathbf{q}_i$$和key $$\mathbf{k}_j$$间的非向量attention得分 |
| $$\mathbf{P} \in \mathbb{R}^{L \times d}$$                     | 位置编码矩阵，其中第$$i$$-th行$$\mathbf{p}_i$$是输入$$\mathbf{x}_i$$的位置编码 |

## Attention and Self-Attention

*Attention* 是一个神经网络中的机制，它表明模型可通过选择性地注意一个给定的数据的集合来学习做出预测。attention的总量被学习到的权重量化，因此输出常常以一个加权平均数的形态给出。

*Self-attention* 是attention机制的一种，模型利用同一个样本的其他部分的观测来为一个数据样本的一个部分做出预测。从概念上说，这跟 [non-local means](https://en.wikipedia.org/wiki/Non-local_means)非常接近。同样请注意，self-attention是具有交换不变形性permutation-invariant的；也就是说，它是对于集合的操作。

attention 和 self-attention都有许多种形式。Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) 依赖于*scaled dot-product attention*：给定一个query矩阵$$\mathbf{Q}$$，一个key矩阵$$\mathbf{K}$$和一个value矩阵$$\mathbf{V}$$，输出是value向量的加权和，其中分配给每个value的权重都由query和对应的key的点乘结果决定：
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q} {\mathbf{K}}^\top}{\sqrt{d_k}})\mathbf{V}
$$
对于一个query向量和一个key向量$$\mathbf{q}_i, \mathbf{k}_j \in \mathbb{R}^d$$（query矩阵和key矩阵中的行向量），我们有非向量得分：
$$
a_{ij} = \text{softmax}(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}})
= \frac{\exp(\mathbf{q}_i {\mathbf{k}_j}^\top)}{ \sqrt{d_k} \sum_{r \in S_i} \exp(\mathbf{q}_i {\mathbf{k}_r}^\top) }
$$
其中，$$S_i$$是与第$$i$$-th个query互相关的key位置的集合。

如果你感兴趣，可以查看我关于其他attention类型的 [博客](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#a-family-of-attention-mechanisms)。

## Multi-Head Self Attention

*multi-head self-attention*模块是Transformer中的关键成分。多头机制将输入分割成小块，然后在每个子空间中并行计算 scaled dot-product attention ，而不是仅仅计算一次attention。独立attention输出被简单地连结，并被线性转换为想要的维度。
$$
\begin{aligned}
\text{MultiHeadAttention}(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v) &= [\text{head}_1; \dots; \text{head}_h] \mathbf{W}^o \\ 
\text{where head}_i &= \text{Attention}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{X}_k\mathbf{W}^k_i, \mathbf{X}_v\mathbf{W}^v_i)
\end{aligned}
$$
其中$$[.;.]$$是连结操作。$$\mathbf{W}^q_i, \mathbf{W}^k_i \in \mathbb{R}^{d \times d_k/h}, \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$$是权重矩阵，将大小为$$L \times d$$的输入嵌入映射成query，key和value矩阵。$$\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$$是输出线性变换。应在训练过程中学习所有权重。

![image-20210415153858332](/assets/img/typora-user-images/image-20210415153858332.png)

## Transformer

**Transformer** （在下文中指的是"vanilla Transformer"，和其他增强版本分别开来； [Vaswani, et al., 2017](https://arxiv.org/abs/1706.03762)）模型有着encoder-decoder结构，这种结构在许多[NMT](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#born-for-translation)模型中用到。之后我们会讲述decoder-only Transformer，它在语言建模任务如[GPT and BERT](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#openai-gpt)中取得了非常好的表现。

### Encoder-Decoder Architecture

**encoder**生成基于attention的表示，它能够在一个很大的context中定位一个特定的信息片。它包括6个相同的模块的堆叠，每个模块都有两个子模块，一个是*multi-head self-attention*层，一个是*point-wise*全链接前向网络。通过point-wise，可将相同的线性变化（权重也是相同的）用于序列中的每个元素上。这种机制也可视为卷积核尺寸为1的卷积层。每个子模块都有一个residual connection和一个层标准化。所有的子模块都输出相同维度$$d$$的数据。

Transformer **decoder**的作用是从编码过的表示中回收信息。这种结构与encoder的结构很类似，但是decoder在每个相同并重复的模块中都包含两个multi-head attention子模块，而不是一个。

![image-20210415154851718](/assets/img/typora-user-images/image-20210415154851718.png)

### Positional Encoding

由于self-attention操作是具有交换不变性的，因此使用合适的**positional encoding**来为模型提供*位置信息 order information*是很重要的。位置编码$$\mathbf{P} \in \mathbb{R}^{L \times d}$$与输入嵌入有着相同维度，因此它可直接被加入输入。vanilla Transformer有两种编码类型：

1. *Sinusoidal positional encoding*：给定表征位置token position $$i=1,\dots,L$$和维度$$\delta=1,\dots,d$$：
   $$
   \text{PE}(i,\delta) = 
   \begin{cases}
   \sin(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
   \cos(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
   \end{cases}
   $$
   如此，位置编码的每个维度都对应于不同维度中的不同波长的正弦曲线，从$$2\pi$$到$$10000 \cdot 2\pi$$。

   ![image-20210415155607406](/assets/img/typora-user-images/image-20210415155607406.png)

2. *Learned positional encoding*：它为每个元素都分配一个学习到的列向量，列向量编码了元素的*绝对 absolute*位置 ([Gehring, et al. 2017](https://arxiv.org/abs/1705.03122))。

### **Quick Follow-ups**

[Al-Rfou et al. (2018)](https://arxiv.org/abs/1808.04444) 在vanilla Transformer的基础上，增添了一系列辅助损失在character-level语言建模任务中训练深度Transformer模型，其效果比LSTMs的更好。它使用了如下几种辅助任务：

- 在每个序列的尾端，它并不是只产生一个预测，而是让每个*中间位置 immediate position*也产生一个正确预测，迫使模型在给定小contexts的情况下也能预测（例如，在context窗口中最开始的那几个tokens）。
- 每个中间Transformer层也都被用来做出预测。随着训练的进展，向低层赋予的权重变得越来越小。
- 序列中的每个位置都能够预测多个目标，例如，对于未来tokens的两个或多个预测。

![image-20210415162023427](/assets/img/typora-user-images/image-20210415162023427.png)

## Adaptive Computation Time (ACT)

**Adaptive Computation Time** (short for **ACT**; [Graves, 2016](https://arxiv.org/abs/1603.08983)) 是在RNN中动态计算需要多少计算步的机制。 [这里](https://distill.pub/2016/augmented-rnns/#adaptive-computation-time)有一篇很好的关于ACT的教程。

假设我们有一个RNN模型$$\mathcal{R}$$，它由输入权重$$W_x$$，带有参数的状态转移函数$$\mathcal{S}(.)$$，一系列输出权重$$W_y$$和输出偏置$$b_y$$。给定一个输入序列$$(x_1, \dots, x_L)$$，输出序列$$(y_1, \dots, y_L)$$由下列公式计算：
$$
s_t = \mathcal{S}(s_{t-1}, W_x x_t), \quad y_t = W_y s_t + b_y\quad\text{for }t=1, \dots, L
$$
ACT使得上述RNN能够展示每个输入元素上的不同步数。多个计算步产生了一系列中间状态$$(s_t^1, \dots, s_t^{N(t)})$$和输出$$(y_t^1, \dots, y_t^{N(t)})$$——它们拥有同样的状态转移函数$$\mathcal{S}(.)$$，以及同样的输出权重$$W_y$$和偏置$$b_y$$：
$$
\begin{aligned}
s_t^0 &= s_{t-1} \\
s_t^n &= \mathcal{S}(s_{t}^{n-1}, x_t^n) = \mathcal{S}(s_{t}^{n-1}, x_t + \delta_{n,1}) \text{ for } n=1, \dots, N(t)\\
y_t^n &= W_y s_t^n + b_y
\end{aligned}
$$
其中$$\delta_{n,1}$$是一个二元flag，它表示输入步是否增加。

计算步数量$$N(t)$$由一个额外的sigmoid 暂停单元$$h$$决定，它与权重矩阵$$W_h$$和偏置$$b_h$$相关。$$h$$输出一个对于$$t$$-th输入元素的第$$n$$个中间步上的暂停概率$$p_t^n$$：
$$
h_t^n = \sigma(W_h s_t^n + b_h)
$$
为了使在一单步后暂停计算，ACT引入了一个小常数$$\epsilon$$（例如，0.01），则只要积累概率达到$$1-\epsilon$$以上，则计算停止。
$$
\begin{aligned}
N(t) &= \min(\min\{n': \sum_{n=1}^{n'} h_t^n \geq 1 -\epsilon\}, M) \\
p_t^n &= \begin{cases}
h_t^n & \text{if }n < N(t) \\
R(t) = 1 - \sum_{n=1}^{N(t)-1} h_t^n & \text{if }n= N(t)\\
\end{cases}
\end{aligned}
$$
其中$$M$$是设定的中间步数量上限。

最终状态和输出是mean-field更新：
$$
s_t = \sum_{n=1}^{N(t)} p_t^n s_t^n,\quad y_t = \sum_{n=1}^{N(t)} p_t^n y_t^n
$$
![image-20210415164516334](/assets/img/typora-user-images/image-20210415164516334.png)

为了避免每个输入上多余的pondering，ACT在损失函数中增加了一个*ponder cost* $$\mathcal{P}(x) = \sum_{t=1}^L N(t) + R(t)$$，使得中间计算步数量趋向于更少。

## Improved Attention Span

增加attention跨度的目的是创造出可用于self-attention的更长、更有效率并且更加灵活的context。

### Longer Attention Span (Transformer-XL)

vanilla Transformer的attention跨度是固定并且有限的。模型在每次更新中都只能注意到同一个segment中的其他元素，没有信息能够流过分开的固定长度的segments。

*context segmentation*会造成几个问题：

- 模型不能捕捉较长项依赖；
- 如果不给出context或者context很小，那么预测每个segment中前几个tokens会很难；
- evaluation开销太大。只要segment向右移动一位，新的segment就得从头重新处理，即使有许多重叠的tokens。

**Transformer-XL** ([Dai et al., 2019](https://arxiv.org/abs/1901.02860); “XL” means “extra long”)通过两个主要改动解决了context分割问题：

1. 重用segments间的隐状态；
2. 采用一个新的、对于重用状态很适合的位置编码。

#### Hidden State Reuse

通过持续使用来自先前segments的隐状态，向模型中引入segments间的循环连接。

![image-20210415165606372](/assets/img/typora-user-images/image-20210415165606372.png)

我们将模型中第$$\tau+1$$-th个segment在第$$n$$-th层中的隐状态记作$$\mathbf{h}_{\tau+1}^{(n)} \in \mathbb{R}^{L \times d}$$。除了同一个segment在上一层中的隐状态$$\mathbf{h}_{\tau+1}^{(n-1)}$$，它还依赖于上一个segment 在同一层中的隐状态$$\mathbf{h}_{\tau}^{(n)}$$。通过与这些先前的隐状态中的信息相结合，模型便可将attention跨度在多个segments上向更久远的时间延伸。
$$
\begin{aligned}
\color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} &= [\text{stop-gradient}(\mathbf{h}_{\tau}^{(n-1)}) \circ \mathbf{h}_{\tau+1}^{(n-1)}] \\
\mathbf{Q}_{\tau+1}^{(n)} &= \mathbf{h}_{\tau+1}^{(n-1)}\mathbf{W}^q \\
\mathbf{K}_{\tau+1}^{(n)} &= \color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} \mathbf{W}^k \\
\mathbf{V}_{\tau+1}^{(n)} &= \color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} \mathbf{W}^v \\
\mathbf{h}_{\tau+1}^{(n)} &= \text{transformer-layer}(\mathbf{Q}_{\tau+1}^{(n)}, \mathbf{K}_{\tau+1}^{(n)}, \mathbf{V}_{\tau+1}^{(n)})
\end{aligned}
$$
请注意，key和value都依赖于延伸了的隐状态，然而query仅仅利用当前步的隐状态。连结操作$$[. \circ .]$$是对于序列长度维度进行的。

#### **Relative Positional Encoding**

为了适应这种attention跨度的新形势，Transformer-XL提出了一种新类型的位置编码。如果使用和vanilla Transformer 相同的方法，编码绝对位置，则之前和当前的segments都会被赋予相同的编码，这是我们不希望看到的。

为了使位置信息流与segments保持一致，Transformer-XL编码了*relative*位置，因为仅了解位置偏移offset对于做出好的预测来说就已经足够了，比如，$$i-j$$，key向量$$\mathbf{k}_{\tau, j}$$和它的query $$\mathbf{1}_{\tau, i}$$之间的位置偏移。

如果我们忽略尺度因子$$1/\sqrt{d_k}$$和softmax中的标准化项，但是加入位置编码，我们就能写出位置位$$i$$的query和位置为$$j$$的key之间的attention得分：
$$
\begin{aligned}
a_{ij} 
&= \mathbf{q}_i {\mathbf{k}_j}^\top = (\mathbf{x}_i + \mathbf{p}_i)\mathbf{W}^q ((\mathbf{x}_j + \mathbf{p}_j)\mathbf{W}^k)^\top \\
&= \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top
\end{aligned}
$$
Transformer-XL将上面的四项重新参数化为：
$$
a_{ij}^\text{rel} = 
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{content-based addressing} + 
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{content-dependent positional bias} + 
\underbrace{ \color{red}{\mathbf{u}} \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{global content bias} + 
\underbrace{ \color{red}{\mathbf{v}} \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{global positional bias}
$$

- 将$$\mathbf{p}_j$$替换为位置编码$$\mathbf{r}_{i-j} \in \mathbf{R}^{d}$$；
- 将两个不同项中的$$\mathbf{p}_i\mathbf{W}^q$$替换为两个可训练的参数$$\mathbf{u}$$（对于内容content）和$$\mathbf{v}$$（对于位置location）；
- 将$$\mathbf{W}^k$$分割为两个矩阵，对于内容信息的$$\mathbf{W}^k_E$$和对于位置信息的$$\mathbf{W}^k_R$$

### Adaptive Attention Span

Transformer的一个关键优势在于它能够捕捉长距离的依赖。根据context，模型有可能对于更远处的信息关注较多；或者某个attention头可能有着与别的头不同的attention模式。如果attention跨度能够灵活适应context长度，并且仅仅在需要的时候回头关注较远处，这可能会对降低计算和内存开销有所帮助，更能支持模型中的更长的最大context尺寸。

这就是**Adaptive Attention Span**被创造出来的动因。[Sukhbaatar, et al., (2019)](https://arxiv.org/abs/1905.07799) 提出了一个寻求最优attention跨度的self-attention机制。作者假设不同的attention头会在相同的context窗口中分配不同的分数，因此单个头上的最优跨度应该被单独训练。

![image-20210415201604687](/assets/img/typora-user-images/image-20210415201604687.png)

给定第$$i$$-th个token，我们需要计算出这个token和其他在位置$$j\in S_i$$上的keys之间的attention权重，其中$$S_i$$定义了第$$i$$-th个token的context窗口。
$$
\begin{aligned}
e_{ij} &= \mathbf{q}_i {\mathbf{k}_j}^\top \\ 
a_{ij} &= \text{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{r=i-s}^{i-1} \exp(e_{ir})} \\
\mathbf{y}_i &= \sum_{r=i-s}^{i-1}a_{ir}\mathbf{v}_r = \sum_{r=i-s}^{i-1}a_{ir}\mathbf{x}_r\mathbf{W}^v
\end{aligned}
$$
为了得到有效的、可调整的attention span，添加一个*soft mask function* $$m_z$$ 来控制，它将query和key之间的距离映射成为$$[0,1]$$之间的一个值。 $$m_z$$的参数为$$z \in [0, s]$$，$$z$$应被训练得到：
$$
m_z(x) = \text{clamp}(\frac{1}{R}(R+z-x), 0, 1)
$$
其中$$R$$是超参数，它定义了$$m_z$$的软度softness。

![image-20210415202203229](/assets/img/typora-user-images/image-20210415202203229.png)

应用此soft mask函数到attention权重中的softmax项上：
$$
a_{ij} = \frac{m_z(i-j)\exp(s_{ij})}{\sum_{r=i-s}^{i-1}m_z(i-r) \exp(s_{ir})}
$$
上式中，$$z$$是可微的，因此它可与模型中其他部分一起被训练。参数$$z^{(i)}, i=1, \dots, h$$在每个头上被分开学习，损失函数在$$\sum_{i=1}^h z^{(i)}$$上有一个额外的L1惩罚项。

利用[Adaptive Computation Time](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html#adaptive-computation-time-act)，AAS可被进一步提升，以拥有灵活的attention跨度长度，便可动态适应当前输入。一个attention头在时间$$t$$时的跨度参数$$z_t$$是一个sigmoidal函数，$$z_t = S \sigma(\mathbf{v} \cdot \mathbf{x}_t +b)$$，其中向量$$\mathbf{v}$$和偏置因子$$b$$和其他参数一起被学习。

带有AAS的Transformer的实验中， [Sukhbaatar, et al. (2019)](https://arxiv.org/abs/1905.07799)发现了一个大体趋势：较低的层不需要很长的attention跨度，然而在更高层中的一些attention头却可能使用极其长的跨度。AAS也对于减少FLOPS的数量极有帮助，尤其是在有着许多attention层和很长的context的大模型中。

### Localized Attention Span (Image Transformer)

最初并且最流行的Transformer 应用领域就是语言建模。文本序列是有着很清楚的时序顺序的一维向量，因此attention跨度会随着context大小的增长而线性增长。

然而，如果我们想在图像上使用Transformer，如何定义context的范围scope或顺序则是不清楚的。**Image Transformer** ([Parmer, et al 2018](https://arxiv.org/abs/1802.05751))利用了一种与Transformer框架中的序列建模sequence modeling很相似的一种图像生成方法。并且，Image Transformer将self-attention的跨度限制在了*局部 local*邻域上，因此模型可增大以并行处理更多图像，并且同时保持似然损失可追踪。

encoder-decoder结构仍可用于image-conditioned生成：

- encoder生成一个源图像的语境话的contextualized、每像素通道per-pixel-channel的表示；
- decoder*自动退化地 autoregressively*生成一个输出图像，每个时间步、每个像素上生成一个通道。

我们将（即将被生成的）当前像素的表示标记为query $$\mathbf{q}$$。其他可能会用于计算$$\mathbf{q}$$的表示的位置是key向量$$\mathbf{k}_1,\mathbf{k}_2,...$$，它们一起形成了一个记忆矩阵memory $$\mathbf{M}$$。$$\mathbf{M}$$的视野定义了像素query$$\mathbf{q}$$的context窗口。

Image Transformer引入了两种类型的局部化$$\mathbf{M}$$，如下图所示。

![image-20210415205346581](/assets/img/typora-user-images/image-20210415205346581.png)

1. *1D Local Attention*：输入图像以 [raster scanning](https://en.wikipedia.org/wiki/Raster_scan#Scanning_pattern) 顺序被扁平化，即从左向右、向上到下。将线性化后的图像分成不重叠的query块。context窗口包含同一个query块 $$\mathbf{q}$$中的像素，以及在此query块之前生成的有着固定数量的额外像素。
2. *2D Local Attention*：图像被分为多个不重叠的长方形query块。query像素与同一个memory块中的所有其他像素互相关。为了确保在左上方的像素也能有可靠的context窗口，memory块会以固定总数被分别延拓至上方、左边和右边。

## Less Time and Memory Cost

本章将引入几种 Transformer上的改进，以减少计算时间和内存开销。

### Sparse Attention Matrix Factorization (Sparse Transformers)

vanilla Transformer的计算和内存成本会以四倍的速度随着序列长度增长，因此很难应用在长序列中。

**Sparse Transformer** ([Child et al., 2019](https://arxiv.org/abs/1904.10509))引入了一种*factorized self-attention*，通过稀疏矩阵因数分解，使得在长至16384的序列上训练层数以百计的稠密attention网络成为了可能。

给定一个attention连接性模式connectivity pattern $$\mathcal{S} = \{S_1, \dots, S_n\}$$，其中$$S_i$$保存了一个与第$$i$$-th个query互相关的key位置的集合。
$$
\begin{aligned}
\text{Attend}(\mathbf{X}, \mathcal{S}) &= \Big( a(\mathbf{x}_i, S_i) \Big)_{i \in \{1, \dots, L\}} \\
\text{ where } a(\mathbf{x}_i, S_i) &= \text{softmax}\Big(\frac{(\mathbf{x}_i \mathbf{W}^q)(\mathbf{x}_j \mathbf{W}^k)_{j \in S_i}^\top}{\sqrt{d_k}}\Big) (\mathbf{x}_j \mathbf{W}^v)_{j \in S_i}
\end{aligned}
$$
请注意，即使$$S_i$$的大小是不固定的，$$a(\mathbf{x}_i, S_i)$$的尺寸总是$$d_v$$，因此$$\text{Attend}(\mathbf{X}, \mathcal{S})\in \mathbf{R}^{l\times d_v}$$。

在自主退化模型中，attention跨度被定义为$$S_i = \{j: j \leq i\}$$，它允许每个token都与过往的所有位置互相关。

在因数分解后的self-attention中，集合$$S_i$$被分解为一个依赖*树*，因此对于每对$$(i, j)$$（其中$$j \leq i$$），都有一条将$$i$$连接回$$j$$的路径，$$i$$可直接或非直接地与$$j$$互相关。

准确地说，集合$$S_i$$被分解为$$p$$个*不重叠*的子集，其中第$$m$$-th个子集被表示为$$A^{(m)}_i \subset S_i, m = 1,\dots, p$$。因此输出位置$$i$$和任何$$j$$之间的路径都有一个最大长度$$p+1$$。例如，如果$$(j, a, b, c, \dots, i)$$是$$i$$与$$j$$之间的路径索引，则我们有$$j \in A_a^{(1)}, a \in A_b^{(2)}, b \in A_c^{(3)}, \dots$$，如此这样下去。

**Sparse Factorized Attention**提出了两种因数分解attention。通过图10中的2D图像输入来理解这些概念会更简单些。

![image-20210415220300873](/assets/img/typora-user-images/image-20210415220300873.png)

1.  *Strided* Attention：步长为$$\ell \sim \sqrt{n}$$。这种类型对于图像数据很合适，因为结构通过步长来排列。在此图像中，每个像素都可与光栅扫描顺序raster scanning order（覆盖了图像的全部宽度）中先前的$$\ell$$像素互相关，然后，这些像素与同一列中的其他像素互相关（由另一个attention连接性子集定义）。

$$
\begin{aligned}
A_i^{(1)} &= \{ t, t+1, \dots, i\} \text{, where } t = \max(0, i - \ell) \\
A_i^{(2)} &= \{j: (i-j) \mod \ell = 0\}
\end{aligned}
$$

2. *Fixed* Attention：tokens的一个小集合概括了先前位置，并且将信息传播给所有未来位置。

$$
\begin{aligned}
A_i^{(1)} &= \{j: \lfloor \frac{j}{\ell} \rfloor = \lfloor \frac{i}{\ell} \rfloor \} \\
A_i^{(2)} &= \{j: j \mod \ell \in \{\ell-c, \dots, \ell-1\} \}
\end{aligned}
$$

​		其中$$c$$是超参数。如果$$c=1$$，它就限制住了表示，然而许多表示都依赖于一些位置。此论文对于$$\ell \in \{ 128, 256 \}$$选择$$c\in \{ 8, 16, 32 \}$$。

**Use Factorized Self-Attention in Transformer**

在Transformer结构中，有三种方法来使用稀疏因数分解attention模式sparse factorized attention patterns：

1. 对于单个剩余residual块分配一个attention，然后interleave它们，$$\text{attention}(\mathbf{X}) = \text{Attend}(\mathbf{X}, A^{(n \mod p)}) \mathbf{W}^o$$，其中$$n$$是当前剩余块的索引。
2. 设定一个与那些与所有因数分解后的头都互相关的位置互相关的单个头，$$\text{attention}(\mathbf{X}) = \text{Attend}(\mathbf{X}, \cup_{m=1}^p A^{(m)}) \mathbf{W}^o$$。
3. 使用多头attention机制，但不同于vanilla transformer，每个头都可能采用上述模式中的一种，1或2，这个选择通常表现最好。

稀疏Transformer同样有许多变种以训练多达几百层的Transformer，包括梯度检查点、在反向流动过程中重计算attention&FF层、混合精度训练precision training、实施高效块稀疏efficient block-sparse implementation，等等。可查看[此论文](https://arxiv.org/abs/1904.10509)以研究更多细节。

### Locality-Sensitive Hashing (Reformer)

由**Reformer**模型([Kitaev, et al. 2020](https://arxiv.org/abs/2001.04451)) 提出的改进目的在于解决Transformer中的几个痛点：

- $$N$$层模型的内存是单层模型内存的$$N$$倍大，这是由于我们需要为反向传播存储activations。
- 中间FF层都相当大；
- 长度为$$L$$的序列上的attention在内存和时间上的复杂度为$$O(L^2)$$。

Reformer主要提出了两个改进：

1. 将dot-product attention替换为*locality-sensitive hashing (LSH) attention*，这将复杂度从$$O(L^2)$$减小至$$O(L\text{log}L)$$。
2. 将标准剩余块用*reversible residual layers*替换，它只允许在训练过程中对activations存储一次，而不是$$N$$次（例如，与层数成比例）。

**Locality-Sensitive Hashing Attention**

[attention formula](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html#attention-and-self-attention)的$$\mathbf{Q} \mathbf{K}^\top$$部分中，我们只对最大元素感兴趣，因为只有大元素才在softmax后贡献较大。对于每个$$\mathbf{q}_i \in \mathbf{Q}$$，我们寻找$$\mathbf{K}$$中与$$\mathbf{q}_i$$最接近的行向量。为了在高维空间中快速找到最近邻，Reformer将 [Locality-Sensitive Hashing (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)添加到attention机制当中。

如果哈希方法$$x \mapsto h(x)$$保存了数据点之间的距离信息，则它是*locality-sensitive*的，因此邻近的向量获得了相似哈希值，距离较远的向量获得很不同的哈希值。Reformer采用了一种哈希方法：给定一个固定随机矩阵$$\mathbf{R} \in \mathbb{R}^{d \times b/2}$$（其中$$b$$是超参数），哈希函数是$$h(x) = \arg\max([xR; −xR])$$。

![image-20210416134445302](/assets/img/typora-user-images/image-20210416134445302.png)

在LSH attention中，一个query只能与同一个哈希bucket中的位置互相关，$$S_i = \{j: h(\mathbf{q}_i) = h(\mathbf{k}_j)\}$$。它在后续步骤中执行，如图11所示：

- (1) 完整（full）attention的attention矩阵通常是稀疏的。
- (2) 利用LSH，我们能够将keys和queries分成根据它们的哈希bucket来排列的形式。
- (3) 设置$$\mathbf{Q} = \mathbf{K}$$（准确地说，$$\mathbf{k}_j = \mathbf{q}_j / \|\mathbf{q}_j\|$$），这样在同一个bucket中的keys和queries的数量就相等，这使得batching更加容易。有趣的是，这个“shared-QK”设置并不影响Transformer的性能。
- (4) 将batching应用于$$m$$个连续queries被合成一组的块。

![image-20210416143814599](/assets/img/typora-user-images/image-20210416143814599.png)

**Reversible Residual Network**

另一个由Reformer带来的改善是使用*reversible residual layers* ([Gomez et al. 2017](https://arxiv.org/abs/1707.04585))。使用reversible residual network的原因是想要设计这样一个结构，其中任何给定层中的activations都可从下一层中的activations上恢复出来。因此我们可通过在反向传播中重计算activations而不是存储所有的activations来节省内存。

给定一层$$x \mapsto y$$，普通的residual层是这样操作：$$y = x + F(x)$$，但是reversible层将输入和输出都分为一对儿：$$(x_1, x_2) \mapsto (y_1, y_2)$$，然后执行如下操作：
$$
y_1 = x_1 + F(x_2),\; y_2 = x_2 + G(y_1)
$$
reversing很简单：
$$
x_2 = y_2 - G(y_1), \; x_1 = y_1 − F(x_2)
$$
Reformer将此思路通过将attention$$(F)$$和前项层$$(G)$$在一个reversible net块中结合的方式，应用于Transformer：
$$
Y_1 = X_1 + \text{Attention}(X_2), \; Y_2 = X_2 + \text{FeedForward}(Y_1)
$$
内存开销可进一步通过连结前项计算来减少：
$$
Y_2 = [Y_2^{(1)}; \dots; Y_2^{(c)}] = [X_2^{(1)} + \text{FeedForward}(Y_1^{(1)}); \dots; X_2^{(c)} + \text{FeedForward}(Y_1^{(c)})]
$$

## Make it Recurrent (Universal Transformer)

**Universal Transformer** ([Dehghani, et al. 2019](https://arxiv.org/abs/1807.03819)) 结合了Transformer中的self-attention和RNN中的循环机制，目的是引入Transformer的长期全局接受域和RNN的learned inductive biases。

Universal Transformer可通过动态调节 [adaptive computation time](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html#adaptive-computation-time-act)来计算步骤数目，而不是走过固定数目的层。如果我们将步骤数固定住，则Universal Transformer就等同于在所有层中共享参数的多层Transformer了。

在一个较高的层中，universal transformer可被视作一个循环函数，以学习每个token的隐状态表达。循环函数在token位置中并行进化，位置间的信息在self-attention中共享。

![image-20210416150204688](/assets/img/typora-user-images/image-20210416150204688.png)

给定一个长度为$$L$$的输入序列，Universal Transformer在第$$t$$步以一个可调整的步骤数来迭代更新表示$$\mathbf{H}^t \in \mathbb{R}^{L \times d}$$。在第0步，将$$\mathbf{H}^0$$初始化成与输入嵌入矩阵相同。所有位置都在多头self-attention机制中被并行处理，然后经过一个循环转移函数。
$$
\begin{aligned}
\mathbf{A}^t &= \text{LayerNorm}(\mathbf{H}^{t-1} + \text{MultiHeadAttention}(\mathbf{H}^{t-1} + \mathbf{P}^t) \\
\mathbf{H}^t &= \text{LayerNorm}(\mathbf{A}^{t-1} + \text{Transition}(\mathbf{A}^t))
\end{aligned}
$$
其中$$\text{Transition}(.)$$是一个 [separable convolution](https://arxiv.org/abs/1610.02357)或全链接NN，包含两个position-wise（也就是单独应用于$$\mathbf{A}^t$$的每行）仿射变换+一个ReLU。

位置编码$$\mathbf{P}^t$$使用sinusoidal位置信号，但带有一个额外的时间维度：
$$
\text{PE}(i, t, \delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) \oplus \sin(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) \oplus \cos(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}
$$
![image-20210416150923480](/assets/img/typora-user-images/image-20210416150923480.png)

在Universal Transformer的一个适应性版本中，循环步数$$T$$由[ACT](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html#adaptive-computation-time-act)动态决定。每个位置都具有一个动态ACT暂停机制。一旦某个per-token循环块暂停，它就停止进行recurrent更新，仅是简单地将当前值拷贝至下一步，直到所有块暂停或模型达到最大步数限制。

## Stabilization for RL (GTrXL)

self-attention机制避免了将整个过去状态压缩进一个固定尺寸的隐状态，因此不会像RNNs那样经历梯度爆炸或消失的问题。RL任务一定能从这些特性中获益。然而，在监督学习中训练Transformer都是相当困难的，更不用说在RL任务中了。毕竟，让它稳定一个LSTM智能体并且训练是相当有挑战性的。

**Gated Transformer-XL** (**GTrXL**; [Parisotto, et al. 2019](https://arxiv.org/abs/1910.06764)) 是在RL中使用Transformer的一个尝试性方法。GTrXL通过在[Transformer-XL](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html#longer-attention-span-transformer-xl)上进行两个改动，成功地稳定了训练过程：

1. 层标准化仅应用于residual模块中的输入流上，而不是在shortcut流上。对于此重排列的关键好处是，它使得原始输入从第一层流至最后一层。

2. residual连接被替换为GRU-style (Gated Recurrent Unit; [Chung et al., 2014](https://arxiv.org/abs/1412.3555)) *gating*机制。
   $$
   \begin{aligned}
   r &= \sigma(W_r^{(l)} y + U_r^{(l)} x) \\
   z &= \sigma(W_z^{(l)} y + U_z^{(l)} x - b_g^{(l)}) \\
   \hat{h} &= \tanh(W_g^{(l)} y + U_g^{(l)} (r \odot x)) \\
   g^{(l)}(x, y) &= (1-z)\odot x + z\odot \hat{h}
   \end{aligned}
   $$

gating函数参数被显式初始化到接近于identity map的程度，这就是为什么式中有一$$b_g$$项。$$b_g > 0$$对于学习加速非常有帮助。

![image-20210416152506056](/assets/img/typora-user-images/image-20210416152506056.png)
