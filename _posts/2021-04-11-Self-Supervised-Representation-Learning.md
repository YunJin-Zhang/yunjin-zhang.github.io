---
title: Self-Supervised Representation Learning
layout: post
categories: deep-learning
tags: deep-learning self-supervised generative-model object-recognition
date: 2021-04-11 15:50
excerpt: Self-Supervised Representation Learning
---

# Self-Supervised Representation Learning
本文翻译自lilianweng的博客[Self-Supervised Representation Learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)

> 尽管learning是通过监督学习的方式实现的，但自监督学习为更好地利用未标记数据打开了一扇大门。此篇博客涵盖了许多关于图象、视频和控制问题的自监督学习任务。

如果有一个任务和足够多的标签，那么监督学习可以很好地将其解决。良好性能通常需要可观数目的标签，但是人工收集标签是非常昂贵的（例如，ImageNet）并且很难被扩展。考虑到未标记数据（例如，互联网上的文本和图象）数目远远超过了人类标记的数据集，那么不去利用它们就显得有点浪费。然而，非监督学习并不容易，并且它的效率通常低于监督学习。

如果我们可以为未标记数据获取免费的标签并且通过监督学习的方式来训练非监督数据集，会如何呢？我们可以通过构建一个特殊形式的监督学习任务以利用余下数据集来预测一个子集达到这个目的。借此，就可以利用到所需输入和标签的全部信息了。这就是*self-supervised learning*.

这个思路被广泛使用在语言建模当中。语言模型的一个常见任务是根据过往序列来预测下一个字词。[BERT](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#bert)添加了另外两个辅助任务，它们都是依赖于自生成标签的。

![image-20210411220434692](/assets/img/typora-user-images/image-20210411220434692.png)

[这里](https://github.com/jason718/awesome-self-supervised-learning)是一个关于自监督学习论文的列表。如果你想了解更多，请点开查看。

请注意，本篇博客并不仅关注NLP/语言建模或生成模型。

{:.table-of-content}
* TOC
{:toc}

## Why Self-Supervised Learning?

自监督学习让我们免费利用大量的数据带来的标签。这个动机相当的直接。创建一个有着干干净净的标签的数据集太贵了，但是未标记数据时时刻刻都在产生。为了利用这部分相对来说特别多的未标记数据，其中一个办法就是将学习目标进行完全地设定，以此便可获得完全来源于数据本身的监督supervision。

*self-supervised task* 也叫做 *pretext task*，它指向一个监督损失函数。然而，我们通常并不在乎这个被发明的任务的最终性能。相反，我们对带有期望的学习到的中间表示较感兴趣，这个表示能够带有很好的语义信息或者结构信息，对实际的downstream任务很有帮助。

比如，我们可能在任意位置旋转图像，并且训练一个模型去识别哪个输入图像是被旋转过的。这个旋转预测任务就是人为构造的，因此实际的精确率accuracy就不重要，就像我们对待辅助任务那样。但是我们希望此模型可以为真实世界中任务学习到高质量的潜在变量，比如构建一个拥有极少已标记样本的目标识别分类器。

广泛地说，所有的生成模型都可被视为自监督的，但是它们的目标各有不同：生成模型专属于创造不同且真实的图象，然而自监督表示学习则大体上更加关注创造好的特征，这些特征对许多任务都有帮助。生成建模并不是本篇博客的重点，但可在[之前的博客](https://lilianweng.github.io/lil-log/tag/generative-model)中查看。

## Image-Based

研究者提出了许多对于图像上的自监督表示学习的思路。一个通用的工作流是：首先在一个或多个pretext任务中用未标记图像训练模型，然后使用模型的一个中间特征层为ImageNet分类任务提供多类别逻辑回归分类器。最终的分类精确度决定了学习后表示learned representation的表现。

近期，有些研究者提出了将训练监督学习和训练自监督pretext任务同时进行并共享权重的方法，它们分别使用已标记数据和未标记数据进行训练，如[Zhai et ak, 2019](https://arxiv.org/abs/1905.03670)和[Sun et al, 2019](https://arxiv.org/abs/1909.11825)。

### Distortion

我们希望图像上小的失真不会影响其原始语义或几何形状。轻微失真的图象还是被视为原始图像，因此我们希望学习到的特征对于失真具有不变性。

**Exemplar-CNN**（[Dosovitskiy et al., 2015](https://arxiv.org/abs/1406.6909)）创造出一种代理surrogate训练数据集，它带有未标记图象块：

1. 从不同图像上的不同位置，采样尺寸为$$32\times 32$$像素的$$N$$个图象块，但是仅限那些具有大梯度的区域，因为这些区域覆盖了目标边缘并且很可能包含了目标或者部分目标。这些小区域就是“*exemplary*”区域。

2. 每个区域都通过应用一系列随机变换而失真（例如，平移，旋转，缩放，等等）。所有失真图象都被视为属于同一个代理类别*same surrogate class*。

3. pretext任务就是去区分一系列代理分类*surrogate class*。我们可以任意创建任意数目的代理类别。

   ![image-20210412122007230](/assets/img/typora-user-images/image-20210412122007230.png)

一个完整图象的**Rotation**旋转（[Gidaris et al. 2018](https://arxiv.org/abs/1803.07728)）是另一种有趣又低廉的改变输入图象并保持语义内容不变的方法。每个输入图像首先被旋转$$90^{\circ}$$的倍数，倍数是任意的，也就是$$[0^{\circ},90^{\circ},180^{\circ},270^{\circ}]$$。训练模型以预测图象被旋转了多少度，因此这是一个四分类问题。

为了辨别具有不同旋转角度的相同图象，模型必须学习到如何辨别出高层次的图象部分，如头部、鼻子、眼睛，以及这些部分的相对位置，而不是局部图案。此pretext任务驱使模型通过此方法来学习目标的语义概念。

![image-20210412122827514](/assets/img/typora-user-images/image-20210412122827514.png)

### Patches

自监督学习任务的第二个分类会从一个图象中抽取数个区域，并让模型预测这些区域之间的关系。

[Doersch et al.(2015)](https://arxiv.org/abs/1505.05192)将pretext任务构造为预测来自于同一个图像的两个任意区域之间的相对位置。模型必须能够理解目标的空间语境spatial context，以辨别两个部分的相对位置。

用于训练的区域通过如下方式采样：

1. 不参考图象内容，随机采样第一块区域。
2. 考虑到第一块区域的中心在一个$$3\times3$$的网格上，则第二块区域就应该从第一块区域的8邻域上采样。
3. 为了避免模型仅仅捕捉到低层级的微不足道的信号，比如链接横跨边界的直线或者匹配局部图案，我们引入了额外的噪声：
   - 在区域间增加间隔；
   - 小震动；
   - 将部分区域随机降采样至总共100个像素，再增采样，为pixelation增加鲁棒性；
   - 将绿色和紫红色向灰色方向移动或随机丢弃三个颜色通道中的两个；
4. 训练此模型，以预测8邻域中哪个位置是被第二块区域选中的，这是一个八分类问题。

![image-20210412130821511](/assets/img/typora-user-images/image-20210412130821511.png)

除了一些不重要讯息如边界图形和质地连续性textures continuing，还发现了另一个有趣并且有点儿令人吃惊的简易解决方案，叫做["*chromatic abberation*"](https://en.wikipedia.org/wiki/Chromatic_aberration)。它的灵感来自于穿过透镜的不同波长的光线的不同焦距。在此过程中，有可能在颜色通道中存在小的偏移offset。因此，模型便可通过简单地比较在两个区域中绿色和紫红色是如何以不同方式分开的，来学着去辨别相对位置。这是个简单的方法，并且无需对图象内容加以改动。我们通过将绿色和紫红色向灰色方向移动或随机丢弃三个颜色通道中的两个来预处理图象，便可避免此现象。

![image-20210412131535548](/assets/img/typora-user-images/image-20210412131535548.png)

既然我们已经在上述任务中在每个图像上都设置了一个$$3\times3$$的网格，那么为什么不将9个区域全部用上，而非仅仅使用2个（这会让任务更加困难）呢？顺着这个思路，[Noroozi&Favaro(2016)](https://arxiv.org/abs/1603.09246)设计了一个**jigsaw puzzle**游戏作为pretext任务：训练模型，将9个被洗牌后的区域放置回其初始位置。

一个CNN可利用共享权重将每个区域独立处理，并且为每个区域的索引（一个预先定义的排列）输出一个概率向量。为了控制jigsaw puzzles的难度，此论文提出，可根据哪个预先定义的排列来洗牌区域，并且配置此模型以预测此排列中所有索引的概率向量。

因为输入区域如何被洗牌都不改变预测的正确顺序。一个可能的加速训练的方案是，使用排列不变性图卷积网络permutation-invariant graph convolutional network GCN，我们便不需要将同一系列的区域洗牌多次，[此论文](https://arxiv.org/abs/1911.00025)的思路与此相同。

![image-20210412132525879](/assets/img/typora-user-images/image-20210412132525879.png)

另一个思路是将“特征”或“视觉原始信息primitives”视作尺度值属性scalar-value，它可在数个区域上进行加和，并在不同区域上进行比较。之后，区域间的关系便可由**counting features**和简单的算术定义（[Noroozi, et al, 2017](https://arxiv.org/abs/1708.06734)）。

此论文考虑了两种变换：

1. Scaling：如果一个图象被放大2倍，则视觉原始信息的数量应保持不变；
2. Tiling：如果一个图象被裁剪至$$2\times2$$网格大小，则视觉原始信息的总数量应该是初始特征总数的4倍。

此模型利用上述的特征计数关系来学习特征encoder$$\phi(.)$$。给定一个输入图象$$\mathbf{x}\in \mathbb{R}^{m\times n\times 3}$$，考虑两种变换符类型：

1. 下采样运算符，$$D:\mathbb{R}^{m\times n\times 3} \rightarrow \mathbb{R}^{\frac{m}{2}\times \frac{n}{2}\times 3}$$：因子为2的下采样；
2. Tiling运算符$$T_i:\mathbb{R}^{m\times n\times 3} \rightarrow \mathbb{R}^{\frac{m}{2}\times \frac{n}{2}\times 3}$$：将第$$i$$-th块tile从图象中的一个$$2\times 2$$的网格中抽取出来。

我们希望学到：
$$
\phi(\mathbf{x})=\phi(D\circ\mathbf{x})=\sum^4_{i=1}\phi(T_i\circ \mathbf{x})
$$
因此，MES损失就是：$$\mathfrak{L}_{feat}=||\phi(D\circ\mathbf{x})-\sum^4_{i=1}\phi(T_i\circ \mathbf{x})||^2_2$$。为了避免不重要讯息$$\phi(\mathbf{x})=\mathbf{0}$$，加入另一个损失项以增大两个不同图象的特征的差异性：$$\mathfrak{L}_{diff}=max(0,\ c-||\phi(D\circ\mathbf{y})-\sum^4_{i=1}\phi(T_i\circ \mathbf{x})||^2_2)$$，其中$$\mathbf{y}$$是另一个不同于$$\mathbf{x}$$的输入图象，并且$$c$$是一个非向量常量。那么最终的损失为：
$$
\mathfrak{L}=\mathfrak{L}_{feat}+\mathfrak{L}_{diff}=||\phi(D\circ\mathbf{x})-\sum^4_{i=1}\phi(T_i\circ \mathbf{x})||^2_2+max(0,\ c-||\phi(D\circ\mathbf{y})-\sum^4_{i=1}\phi(T_i\circ \mathbf{x})||^2_2)
$$
![image-20210412134717903](/assets/img/typora-user-images/image-20210412134717903.png)

### Colorization

**Colorization**可被用作为一个强有力的自监督任务：训练模型，以彩色化一个灰度输入图像；精确地说，此任务是将此图象映射至一个量化了的彩色值输出分布（[Zhang et al.2016](https://arxiv.org/abs/1603.08511)）。

此模型输出的颜色在[CIE Lab* color space](https://en.wikipedia.org/wiki/CIELAB_color_space)当中。Lab* color是为了近似人类视觉而被设计出来的，与此相反的是物理设备上色彩输出的RGB或CMYK模型。

- L\*成分表示人类对于光照的感知；$$L*=0$$是黑色的，$$L*=100$$是白色的；
- a\*成分表示绿色（负）/紫红色（正）值；
- b\*成分表示蓝色（负）/黄色（正）值；

由于colorization问题的多模态本质，在直方图上的所有颜色值的预测概率分布cross-entropy损失都比原颜色值上的L2损失表现要更好。$$b$$颜色空间的bucket size设置为10。

为了平衡普通颜色（通常为低$$b$$值，为普通背景的颜色如云朵、墙面和脏污）和稀有颜色（更倾向于与图象中的关键目标相关），损失函数便会利用一个权重项来重新平衡，此权重项增强了较少见的颜色条的损失。这就如同我们在信息检索模型中同时需要[tf与idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)来为字词打分一样。权重项被构造为：$$(1-\lambda)*Gaussian-kernel-smoothed\ empirical\ probability\ distribution+\lambda*a\ uniform\ distribution $$其中两个分布都是量化$$b$$颜色空间上的。

### Generative Modeling

生成建模中的pretext任务是在学习有意义的隐藏表示时重构原始输入。

**denoising autoencoder**（[Vincent, et al, 2008](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)）学习从另一个版本的图象中恢复出原始图像，这个版本被部分破坏或带有随机噪声。此设计的灵感来源于人类可以非常容易地辨认出带有噪声的图片中的无图，这表明关键视觉特征是可以被提取出来并且与噪声相分离的。可以查阅我的[旧博客](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#denoising-autoencoder)。

训练**context encoder**（[Pathak, et al., 2016](https://arxiv.org/abs/1604.07379)），以填充图象中的丢失部分。设$$\hat{M}$$是一个二进制mask，0表示丢失了的像素，1表示留存的输入像素。此模型利用一个重构L2损失和对抗损失的结合来训练。丢失了的区域由任意形状的mask来定义：
$$
\mathfrak{L}(\mathbf{x})=\mathfrak{L}_{recon}(\mathbf{x})+\mathfrak{L}_{adv}(\mathbf{x})\\
\mathfrak{L}_{recon}(\mathbf{x})=||(1-\hat{M}\odot (\mathbf{x}-E(\hat{M}\odot \mathbf{x}))||^2_2\\
\mathfrak{L}_{adv}(\mathbf{x})=\mathop{max}_\limits{D}\mathbb{E}_\mathbf{x}[log\ D(x)+log\ (1-D(E(\hat{M}\odot \mathbf{x})))]
$$
其中$$E(.)$$是encoder$$D(.)$$是decoder.

![image-20210412145817155](/assets/img/typora-user-images/image-20210412145817155.png)

当我们对图像应用一个mask时，上下文encoder会把部分区域中的所有颜色通道中的信息全部去除。那么如果隐藏一些通道呢？**split-brain autoencoder**（[[Zhang et al., 2017](https://arxiv.org/abs/1611.09842)]）就通过预测剩余颜色通道当中的一部分子通道来进行了此研究。设带有$$C$$个颜色通道的数据张量$$\mathbf{x}\in\mathbb{R}^{h\times w\times |C|}$$为网络的第$$l$$-th层的输入。它被分割成两个分离的部分，$$\mathbf{x}_1\in\mathbb{R}^{h\times w\times |C_1|}$$和$$\mathbf{x}_2\in\mathbb{R}^{h\times w\times |C_2|}$$，其中$$C_1, C_2\subseteq C$$。然后训练这两个子网络来做两个互补的预测：网络$$f_1$$从$$\mathbf{x}_1$$中预测$$\mathbf{x}_2$$，另一个网络$$f_2$$从$$\mathbf{x}_2$$中预测$$\mathbf{x}_1$$。损失是L1损失或者cross entropy（如果颜色值被量化）。

这种分割可在RGB-D或Lab\*颜色空间中发生一次，甚至可在具有任意通道数目的CNN网络中的每一层中发生。

![image-20210412151057232](/assets/img/typora-user-images/image-20210412151057232.png)

生成对抗网络GANs能够学习将简单的隐藏变量映射至任意复杂的数据分布。有研究显示，此种生成模型的隐藏空间可以捕捉数据中的语义变体；例如，当在人脸上训练GAN模型时，一些隐藏变量会与面部表情、眼镜、性别等要素产生链接（[Radford et al., 2016](https://arxiv.org/abs/1511.06434)）。

**Bidirectional GANs**（[Donahue, et al, 2017](https://arxiv.org/abs/1605.09782)）引入了一个额外的encoder$$E(.)$$来学习将输入映射至隐藏变量$$\mathbf{z}$$。discriminator$$D(.)$$在输入数据和隐藏表示的共有空间$$(\mathbf{x},\mathbf{z})$$中进行预测，来区分生成对$$(\mathbf{x},E(\mathbf{x}))$$和真实对$$(G(\mathbf{z}),\mathbf{z})$$。训练模型以优化此目标：$$min_{G,E}\ max_DV(D,E,G)$$，其中generator $$G$$与encoder$$E$$学习生成足够真实的数据和隐藏变量，它们真实的足以迷惑discriminator，与此同时，discriminator$$D$$努力分辨真实的和生成的数据。
$$
V(D, E, G) = \mathbb{E}_{\mathbf{x} \sim p_\mathbf{x}} [ \underbrace{\mathbb{E}_{\mathbf{z} \sim p_E(.\vert\mathbf{x})}[\log D(\mathbf{x}, \mathbf{z})]}_{\log D(\text{real})} ] + \mathbb{E}_{\mathbf{z} \sim p_\mathbf{z}} [ \underbrace{\mathbb{E}_{\mathbf{x} \sim p_G(.\vert\mathbf{z})}[\log 1 - D(\mathbf{x}, \mathbf{z})]}_{\log(1- D(\text{fake}))}) ]
$$
![image-20210412153602221](/assets/img/typora-user-images/image-20210412153602221.png)

### Contrastive Predictive Coding

**Contrastive Predictive Coding (CPC)**（[van den Oord, et al. 2018](https://arxiv.org/abs/1807.03748)）对比预测编码是通过将生成建模问题转换为分类问题的一个高为数据的无监督学习方法。CPC当中的*contrastive loss*和*InfoNCE loss*的灵感来自于[Noise Contrastive Estimation (NCE)](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#noise-contrastive-estimation-nce)，它使用交叉熵损失来衡量模型在一系列无关联“负”样本中分类“未来”表示的性能。这种设计的部分灵感来源于，单模损失（如MSE没有足够的能力，学习一个完整的生成模型的成本太高）。

![image-20210412154225201](/assets/img/typora-user-images/image-20210412154225201.png)

CPC使用一个encoder来压缩输入数据$$z_t = g_\text{enc}(x_t)$$，并利用一个*autoregressive* decoder来学习那些可能在未来预测中被共享的高层级上下文，$$c_t = g_\text{ar}(z_{\leq t})$$。此端到端训练依赖于被NCE引发nce-inspired的对比损失contrastive loss。

在预测未来信息期间，CPC被优化以最大化输入$$x$$和上下文向量$$c$$间的互信息：
$$
I(x; c) = \sum_{x, c} p(x, c) \log\frac{p(x, c)}{p(x)p(c)} = \sum_{x, c} p(x, c)\log\frac{p(x|c)}{p(x)}
$$
CPC建模了一个密度函数以保留$$x_{t+k}$$和$$c_{t}$$之间的互信息，而不是直接建模未来观测$$p_k(x_{t+k} \vert c_t)$$（成本可能会很高）：
$$
f_k(x_{t+k}, c_t) = \exp(z_{t+k}^\top W_k c_t) \propto \frac{p(x_{t+k}|c_t)}{p(x_{t+k})}
$$
其中$$f_k$$可以是未正则化的，线性变换$$W_k^\top c_t$$用来做预测，它对每一步$$k$$都带有一个不同的$$W_k$$矩阵。

给定一个有着$$N$$个随机样本$$X = \{x_1, \dots, x_N\}$$的系列，仅包含一个正样本$$x_t \sim p(x_{t+k} \vert c_t)$$和$$N-1$$个负样本$$x_{i \neq t} \sim p(x_{t+k})$$，则正确分类出正样本（其中$$\frac{f_k}{\sum f_k}$$为预测结果）的交叉熵损失为：
$$
\mathcal{L}_N = - \mathbb{E}_X \Big[\log \frac{f_k(x_{t+k}, c_t)}{\sum_{i=1}^N f_k (x_i, c_t)}\Big]
$$
![image-20210412161442980](/assets/img/typora-user-images/image-20210412161442980.png)

当在图像上使用CPC时（[Henaff, et al. 2019](https://arxiv.org/abs/1905.09272)），predictor网咯应该仅能接触到一个masked特征集，以避免不重要的预测。具体地说：

1. 将每个输入图像分割为一系列互相有重叠的区域，每个区域被一个resnet encoder编码，得到压缩特征向量$$z_{i,j}$$；
2. masked卷积网络利用一个mask来做出预测，则给定的输出神经元的接受域receptive field便能够看见图象中的在其之上的信息。否则，预测问题便会变得琐碎而不重要。可在两个方向上都做出预测（top-down和bottom-up）；
3. 从上下文$$\hat{z}_{i+k, j} = W_k c_{i,j}$$当中对$$z_{i+k, j}$$做出预测。

对比损失对预测进行量化：目标是在一系列负表示$$\{z_l\}$$中正确辨别出目标，这些负表示是从相同图象和同一批次中的其他图象的其他区域中采样出来的：
$$
\mathcal{L}_\text{CPC} 
= -\sum_{i,j,k} \log p(z_{i+k, j} \vert \hat{z}_{i+k, j}, \{z_l\}) 
= -\sum_{i,j,k} \log \frac{\exp(\hat{z}_{i+k, j}^\top z_{i+k, j})}{\exp(\hat{z}_{i+k, j}^\top z_{i+k, j}) + \sum_l \exp(\hat{z}_{i+k, j}^\top z_l)}
$$

### Momentum Contrast

**Momentum Contrast**（**MoCo**; [He et al, 2019](https://arxiv.org/abs/1911.05722)）提供了一个作为*dynamic dictionary look-up*的非监督学习视觉表示的框架。这个字典作为一个很大的数据样本的编码表示的FIFO队列被建构。

![image-20210412164247095](/assets/img/typora-user-images/image-20210412164247095.png)

给定一个存疑样本$$x_q$$，我们便可通过encoder$$f_q:q = f_q(x_q)$$得到一个存疑表示$$q$$。关键样本通过一个momentum encoder $$k_i = f_k (x^k_i)$$进行编码来在字典中产生一个关键表示的列表$$\{k_1, k_2, \dots \}$$。我们假设字典当中有个单独的positive key$$k^+$$可以匹配$$q$$。在此篇论文中，我们利用$$x_q$$的带有不同增强的拷贝版本来创建$$k^+$$。然后，对一个正样本和$$K$$个负样本应用 [InfoNCE](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#contrastive-predictive-coding)对比损失：
$$
\mathcal{L}_q = - \log \frac{\exp(q \cdot k^+ / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)}
$$
其中$$\tau$$是温度超参数。

与另一个相似思路**memory bank** ([Wu et al, 2018](https://arxiv.org/abs/1805.01978v1)) （在数据库中存储了所有数据点的表示，随机采样一个keys集作为负样本）对比，MoCo中的基于队列的字典使得我们能够重利用数据的immediate preceding mini-batches的表示。

使用反向传播来更新关键encoder $$f_k$$是不可追溯的，这是基于队列的字典大小决定的。一个简单的方法是对$$f_k$$和$$f_q$$来使用相同的encoder。不同的是，MoCo提出了使用基于momentum的更新方法，$$f_q$$和$$f_k$$的参数分别被标记为$$\theta_q$$和$$\theta_k$$。
$$
\theta_k \leftarrow m \theta_k + (1-m) \theta_q
$$
其中$$m \in [0, 1)$$是momentum的系数。$$f_k$$的更新上没有梯度流过。

![image-20210412170030729](/assets/img/typora-user-images/image-20210412170030729.png)

**SimCLR** ([Chen et al, 2020](https://arxiv.org/abs/2002.05709)) 提出了一个视觉表示的对比学习的简单框架。它通过最大化同一样本的不同的增强场景之间的agreement来表示视觉输入，利用潜在空间中的对比损失来完成。

![image-20210413125331434](/assets/img/typora-user-images/image-20210413125331434.png)

SimCLR通过以下三步运行：

1. 随机抽样一个样本数为$$n$$的小mini-batch，对每个样本施加两个不同的数据增强操作，得到总数为$$2n$$的增强样本。

$$
\tilde{\mathbf{x}}_i=t(\mathbf{x}),\quad \tilde{\mathbf{x}}_j=t'(\mathbf{x}),\quad
t,\ t'\sim\mathcal{T}
$$

​		其中这两个不同的数据增强运算符，$$t$$ 和 $$t'$$，都是从同一个增强族$$\mathcal{T}$$中抽样的。数据增强包括随		机裁剪、带有随机翻转的缩放、色彩失真、以及高斯模糊；

2. 给定一个正样本对，其他$$2(n-1)$$ 数据点视为负样本。则通过一个base encoder$$f(.)$$ 来生成表示：

$$
\mathbf{h}_i = f(\tilde{\mathbf{x}}_i),\quad \mathbf{h}_j = f(\tilde{\mathbf{x}}_j)
$$

3. 利用cosine相似度$$sim(.,.)$$来定义对比损失。请注意，损失通过$$g(.)$$ 在表示的一个额外投影上计算，而不是在表示$$\mathbf{h}$$上直接计算。但是在downstream任务上，仅用到了表示$$\mathbf{h}$$。

$$
\begin{aligned}
\mathbf{z}_i &= g(\mathbf{h}_i),\quad
\mathbf{z}_j = g(\mathbf{h}_j),\quad
\text{sim}(\mathbf{z}_i, \mathbf{z}_j) = \frac{\mathbf{z}_i^\top\mathbf{z}_j}{\|\mathbf{z}_i\| \|\mathbf{z}_j\|} \\
\mathcal{L}_{i,j} &= - \log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2n} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
\end{aligned}
$$

​		其中$$\mathbf{1}_{[k \neq i]}$$是indicator函数：如果$$k\neq i$$则值为1，否则值为0。$$\tau$$是温度超参数。

![image-20210413131626247](/assets/img/typora-user-images/image-20210413131626247.png)

与SimCLR相比，Moco的优势在于Moco将负样本数量与批次大小分开，但是SimCLR需要很大的批次大小以获得足够的负样本。因此，令人惊讶的是，SimCLR在批次大小减小时，性能会下降。

SimCLR有两种设计，也就是1. MLP投影头MLprojection head；2. 更强强大的数据增强（例如，消减、模糊和更强的色彩失真）。这两种设计都被证明是高效的。将它们与MoCo结合，则得到了**MoCo V2** ([Chen et al, 2020](https://arxiv.org/abs/2003.04297)) ，它并不依赖于大批次样本，也能获得更好的迁移性能。

**BYOL** (“Bootstrap your own latent”; [Grill, et al 2020](https://arxiv.org/abs/2006.07733))生成它达到了一个全新的state-of-the-art结果，并且不需要负样本。它依赖于两个神经网络，一个是 *online* 网络，一个是 *target* 网络，它们与对方交互并且向对方学习。目标网络（参数为$$\xi$$）与在线网络（参数为$$\theta$$）有着相同的结构，但是有着polyak平均化权重：$$\xi \leftarrow \tau \xi + (1-\tau) \theta$$ 。

我们的目标是学习一个可用于downstream任务的presentation $$y$$。参数为$$\theta$$ 的在线网络包括：

- 一个encoder $$f_\theta$$；
- 一个投影因子projector $$g_\theta$$；
- 一个投影因子projector $$q_\theta$$。

目标网络与在线网络的结构相同，但是其参数为$$\xi$$，通过polyak平均 $$\theta$$ 进行更新：$$\xi \leftarrow \tau \xi + (1-\tau) \theta$$。

![image-20210413132844983](/assets/img/typora-user-images/image-20210413132844983.png)

给定一个图象 $$x$$，则通过如下所示方法建构BYOL：

1. 利用两个增强符 $$t \sim \mathcal{T}, t' \sim \mathcal{T}'$$ 来创建两个增强场景：$$v=t(x); v'=t'(x)$$；
2. 将它们编码进表示：$$y_\theta=f_\theta(v), y'=f_\xi(v')$$；
3. 将它们投影进潜在变量：$$z_\theta=g_\theta(y_\theta), z'=g_\xi(v')$$；
4. 在线网络输出映射 $$q_\theta(z_\theta)$$ ；
5. $$q_\theta(z_\theta)$$ 和 $$z'$$ 都被L2正则化，得到 $$\bar{q}_\theta(z_\theta) = q_\theta(z_\theta) / \| q_\theta(z_\theta) \|$$ 和 $$\bar{z'} = z' / \|z'\|$$；
6. 损失 $$\mathcal{L}^\text{BYOL}_\theta$$ 是L2正则化映射 $$\bar{q}_\theta(z_\theta)$$ 和 $$\bar{z'} = z'$$ 间的MSE；
7. 另一个对称损失$$\tilde{\mathcal{L}}^\text{BYOL}_\theta$$ 可通过切换 $$v'$$ 和 $$v$$ 得到；也就是说，将 $$v'$$ 输入至在线网络，将 $$v$$ 输入至目标网络；
8. 最终损失即为 $$ \mathcal{L}^\text{BYOL}_\theta + \tilde{\mathcal{L}}^\text{BYOL}_\theta$$，并且只优化参数 $$\theta$$。

不同于那些最流行的基于对比学习的方法，BYOL并不使用负样本对。大多数bootstrapping方法都依赖于伪标签或簇索引cluster indices，但是BYOL直接bootstrap潜在表示。

令人惊讶并且很有趣的是，即使没有负样本，BYOL仍然运转良好。稍后我会提及 [post by Abe Fetterman & Josh Albrecht](https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html)，他们在复现BYOL时有两个惊人的发现：

1. BYOL在不进行批次正则化时，性能并不比random强；
2. 批次正则化隐式地引发了对比学习的一种形式。

他们相信，对于避免模型崩溃（例如，对于所有数据点，模型都输出全零表示会怎样？）使用负样本是很重要的。批次正则化隐式地引入了对于负样本的依赖性，因为无论一批次中的输入多么相似，这些值都被重分布了（分散在$$\mathcal{N}(0,1)$$ 上），批次正则化因此便可以阻止模型崩溃。如果你主攻此领域，那么我强烈推荐 [此文章](https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html)。

**CURL** ([Srinivas & Laskin, et al. 2020](https://arxiv.org/abs/2004.04136)) 在RL领域中应用了上述思路。CURL通过匹配两个数据增强版本$$o_q$$和$$o_k$$的嵌入embeddings来学习RL任务的视觉表示，这两个数据增强版本是对原始观测 $$o$$ 应用对比损失而产生的。CURL首要依赖于随机剪裁数据增强。将key encoder作为带有权重的momentum encoder实施。The key encoder is implemented as a momentum encoder with weights as EMA of the query encoder weights, same as in [MoCo](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#moco).

RL和监督视觉任务的一个显著差别是，RL依赖于连续帧之间的*时序* 连贯性 *temporal* consistency。因此，CURL在每叠帧上都持续应用增强操作，以保留观测的时序结构信息。

![image-20210413165131989](/assets/img/typora-user-images/image-20210413165131989.png)

## Video-Based

一个视频包含一个语义上相关联的帧序列。附近的帧在时间上很靠近，并且比远处的帧更加互相关。帧的顺序描述了推理和物理逻辑上的确定规则；例如，目标运动应该是光滑的，引力是指向下方的，等等。

一个常见的工作流是：首先用在一个或多个带有未标记视频的pretext任务上训练模型，然后再应用此模型中的一个中间特征层以微调一个简单模型，它用于动作分类、分割或目标追踪的downstream任务。

### Tracking

目标运动会被一个视频帧序列所记录。在附近帧中，同一个目标被捕捉的方式差异通常并不会太大，它通常由目标或相机的微小移动所引发。因此，在附近帧上相同目标上学习来的任何视觉表示在潜在特征空间里都应该很接近。由此思路引发， [Wang & Gupta, 2015](https://arxiv.org/abs/1505.00687)通过在视频中 **tracking moving objects**提出了一个视觉表示的非监督学习方法。

准确地说，存在运动的区域会在一个小时间窗口上被追踪（例如，30帧）。选择第一个区域 $$\mathbf{x}$$ 和最后一个区域$$\mathbf{x}^+$$ 并用它们作为训练数据点。如果我们直接训练模型以最小化这两个区域上的特征向量间的差异，则此模型可能仅仅能够学习将所有东西都映射至同一个值上。为了避免此问题，我们添加了一个额外第三个区域$$\mathbf{x}^-$$ 。模型通过让两个被追踪区域间的距离小于第一个和随机区域间的，在特征空间中学习表示，即 $$D(\mathbf{x}, \mathbf{x}^-)) > D(\mathbf{x}, \mathbf{x}^+)$$，其中$$D(.)$$ 是cosine距离：
$$
D(\mathbf{x}_1, \mathbf{x}_2) = 1 - \frac{f(\mathbf{x}_1) f(\mathbf{x}_2)}{\|f(\mathbf{x}_1)\| \|f(\mathbf{x}_2\|)}
$$
损失函数是：
$$
\mathcal{L}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) 
= \max\big(0, D(\mathbf{x}, \mathbf{x}^+) - D(\mathbf{x}, \mathbf{x}^-) + M\big) + \text{weight decay regularization term}
$$
其中$$M$$ 是非向量常量，它控制两个距离之间的最小间隔；在本论文中，$$M=0.5$$。损失函数在最优情况时，迫使$$D(\mathbf{x}, \mathbf{x}^-) >= D(\mathbf{x}, \mathbf{x}^+) + M$$。

损失函数的这种形式在人脸识别任务中作为 [triplet loss](https://arxiv.org/abs/1503.03832) 被熟知，在此，数据集包含许多从不同相机角度和许多不同的人的图象。$$\mathbf{x}^a$$ 是一个特定人的锚图象，$$\mathbf{x}^p$$ 是同一个人在不同相机角度的正图象，$$\mathbf{x}^n$$ 是一个不同的人的负图象。在嵌入空间embedding space中，$$\mathbf{x}^a$$ 比起$$\mathbf{x}^n$$ 应该更接近于$$\mathbf{x}^p$$。
$$
\mathcal{L}_\text{triplet}(\mathbf{x}^a, \mathbf{x}^p, \mathbf{x}^n) = \max(0, \|\phi(\mathbf{x}^a) - \phi(\mathbf{x}^p) \|_2^2 -  \|\phi(\mathbf{x}^a) - \phi(\mathbf{x}^n) \|_2^2 + M)
$$
此三重损失函数的一个稍微有所不同的形式叫做 [n-pair loss](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective) ，也常被用于机器人任务中学习观测嵌入observation embedding。可在 [后面的章节](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#multi-view-metric-learning)中查看更加相关的内容。

![image-20210413172109290](/assets/img/typora-user-images/image-20210413172109290.png)

相关区域也被追踪到了，并且通过一个两步非监督视觉流 [optical flow](https://en.wikipedia.org/wiki/Optical_flow)方法被提取出来：

1. 获取 [SURF](https://www.vision.ee.ethz.ch/~surf/eccv06.pdf) 兴趣点，利用 [IDT](https://hal.inria.fr/hal-00873267v2/document) 以获得每个SURF点的运动；
2. 给定SURF兴趣点的轨迹。如果flow magnitude大于0.5个像素，将兴趣点分类为运动中点。

在训练过程中，给定一个互相关区域对 $$\mathbf{x}$$ 和$$\mathbf{x}^+$$，在同一个小批次中抽样 $$K$$ 个随机区域 $$\{\mathbf{x}^-\}$$ 以构成 $$K$$ 个训练三重组training triplets。两个epoch后，应用 *hard negative mining* 让训练变得更加困难、更加有效率，也就是，搜索可以最大化损失的随机区域，并利用它们做梯度更新。

### Frame Sequence

很自然地，视频帧都是通过时间顺序来排布的。由此期望——好的表示应该学习帧的正确序列correct sequence——所引发，研究者们提出了数个自监督任务。

其中一个思路是验证帧顺序 **validate frame order** ([Misra, et al 2016](https://arxiv.org/abs/1603.08561))。它的pretext任务是判断一个视频中的帧序列在时间顺序上的排布是否是正确的temporal order（"temporal valid"）。模型需要追踪并推理帧上目标的微小运动，以完成这样的任务。

训练帧都是从高运动high-motion窗口取样的。每次取样五个帧 $$(f_a, f_b, f_c, f_d, f_e)$$，时间戳的顺序是 $$a < b < c < d < e$$。在这五个帧之外，创建一个正元组 $$(f_b, f_c, f_d)$$ 和两个负元组 $$(f_b, f_a,f_d)$$， $$(f_b, f_e,f_d)$$。参数 $$\tau_\max = \vert b-d \vert$$ 控制正训练样本的难度（例如，更高$$\rightarrow$$更难），参数 $$\tau_\min = \min(\vert a-b \vert, \vert d-e \vert)$$ 控制负样本的难度（例如，更低$$\rightarrow$$更难）。

已得证，当作为一预训练步骤被使用时，视频帧顺序验证pretext任务可提升动作识别的downstream任务的性能。

![image-20210413201753183](/assets/img/typora-user-images/image-20210413201753183.png)

O3N(Odd-One-Out Network; [Fernando et al. 2017](https://arxiv.org/abs/1611.06646))中的任务也是基于视频帧序列验证的。它还比上述任务更进一步，它能从多个视频片段中**挑出不正确序列**。

给定 $$N+1$$ 个输入视频片段，其中一个片段的帧被洗牌了，因此其顺序是不正确的，剩下 $$N$$ 个片段的时序排列都是正确的。O3N学习如何预测错误视频片段的位置。在实验中，一共有6个输入片段，每个片段包含6帧。

视频中的**时间箭头** **arrow of time** 中信息量很大，无论是低层级物理现象（例如，引力将物体拉至地面、烟向上升起、水向下流等）还是高层级事件推理（例如，鱼往前游、鸡蛋能够被打破但不能由内而外被反转等）。因此，由此可引发另一个思路，即通过预测时间箭头AoT来学习潜在表示——无论顺放视频还是倒放 ([Wei et al., 2018](https://www.robots.ox.ac.uk/~vgg/publications/2018/Wei18/wei18.pdf))。

分类器应该能捕捉低层级物理现象和高层级语义信息，以此预测时间箭头。T-CAM（Temporal Class-Activation-Map）网络接收 $$T$$ 个组，每个组都包含许多视觉流帧。从每个组的卷积层输出都被连结，并输入至二元逻辑回归以预测时间箭头。

![image-20210413203023735](/assets/img/typora-user-images/image-20210413203023735.png)

有趣的是，在数据集中存在一些人工提示。如果不正确地处理它们，就有可能导致不正确的分类而不是基于真实的视频内容：

- 由于视频压缩的存在，黑色帧可能并不完全是黑色的，反而可能含有时序上的特定信息。因此在实验中应移除黑色帧；
- 过大的相机运动，如垂直平移或放大/缩小，也会为时间箭头提供强信号，但却独立于视频内容。处理阶段应该稳定相机运动。

在被作为预训练步骤使用时，AoT pretext任务可提升动作分类的downstream任务的性能。请注意，我们仍然需要微调。

### Video Colorization

[Vondrick et al. (2018)](https://arxiv.org/abs/1806.09594) 提出了**video colorization**，它是一个自监督学习问题，无需额外微调也能得到可在视频分割和未标记视觉区域追踪上应用的丰富表示。

不同于基于图象的 [colorization](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#colorization)，此处的任务要通过充分利用视频帧上色彩的自然时间连贯性来从一个正常参考彩色帧normal reference frame拷贝色彩到另一个灰度级目标帧（因此，这两个帧不应该在时序上离得太远）。为了持续拷贝色彩，将模型设计成学习在不同帧上持续追钟互相关的像素。

![image-20210413204237597](/assets/img/typora-user-images/image-20210413204237597.png)

这个思路很简单，但很聪明。设 $$c_i$$ 为参考帧中第 $$i-th$$ 个像素的真实颜色，$$c_j$$ 为目标帧中第 $$j-th$$ 个像素的颜色。目标帧中第 $$j-th$$ 个像素的预测颜色 $$\hat{c}_j$$ 是参考帧中所有像素的色彩的加权和，其中权重项表示对于相似度的衡量：
$$
\hat{c}_j = \sum_i A_{ij} c_i \text{ where } A_{ij} = \frac{\exp(f_i f_j)}{\sum_{i'} \exp(f_{i'} f_j)}
$$
其中 $$f$$ 是学习到的相应像素的嵌入embeddings；$$ i'$$ 索引着参考帧中的所有像素。权重项引入了基于注意力的指向机制，如 [matching network ](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html#matching-networks)和 [pointer network](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#pointer-network)。由于完整的相似性矩阵可能会非常大，因此对参考帧和目标帧都应用降采样。$$c_j$$ 和 $$\hat{c}_j$$ 间的categorical交叉熵损失和量化色彩一起被利用，如 [Zhang et al. 2016](https://arxiv.org/abs/1603.08511)所做的那样。

基于参考帧被标记的方法，模型便可用于解决数个基于颜色的downstream任务如时序上的追踪分割或人类姿势。不再需要微调。如图15所示。

![image-20210413205804795](/assets/img/typora-user-images/image-20210413205804795.png)

> 一些常见看法：
>
> - 多种pretext任务相结合提升性能；
> - 更深的网络可改善表示质量；
> - 监督学习的baselines仍然比所有此类方法性能好；

---

## Control-Based

当在现实世界中运行RL策略时，如基于视觉输入来控制一个真实的机器人，完全地追踪状态、获取reward信号或判断目标是否真的达成是很重要的。视觉数据有大量噪声，这些噪声与实际状态并不相关，因此状态的当量便不能从像素级比较中被推理出来。自监督表示学习在学习有用的状态嵌入上有着很大潜力，状态嵌入可直接作为控制策略的输入。

本章讨论的所有情况都是机器人学习领域中的，主要是多种相机视角和目标表示的状态表示。

### Multi-View Metric Learning

度量学习的概念再前叙章节中被多次提及。一个普遍设定是：给定三个样本（锚样本$$s_a$$，正样本$$s_p$$，负样本$$s_n$$），学习到的表示嵌入$$\phi(s)$$可达到如下效果：在潜在空间中，$$s_a$$离$$s_p$$很近，但是离$$s_n$$很远。

**Grasp2Vec** ([Jang & Devin et al., 2018](https://arxiv.org/abs/1811.06964)) 意欲在机器人抓取任务中从任意的、未标记的抓取活动开始，学习一个以目标为中心的表示。意思是，通过以目标为中心，无论环境和机器人看起来如何，只要两个图象包含相似项，则它们就应该映射至相似表示；否则，嵌入就应该分得很开。

![image-20210413212040449](/assets/img/typora-user-images/image-20210413212040449.png)

抓取系统能够分辨它是否将目标移动了，但是不能分辨这是哪个目标。设置相机为拍摄整个场景和被抓取的目标。在训练的早期阶段，抓取机器人抓取任意目标$$o$$，得到一组图像：$$(s_\text{pre}, s_\text{post}, o)$$。

- $$o$$ 是被抓取目标的图象，它被举到相机前方；
- $$s_{pre}$$ 是在抓取之前的场景图象，目标$$o$$在托盘内；
- $$s_{post}$$ 是在抓取之后的相同场景的图象，目标$$o$$不再托盘内。

为了学习以目标为中心的表示，我们期望 $$s_{pre}$$ 嵌入和 $$s_{post}$$ 嵌入之间的差异能捕捉到被去除的目标 $$o$$。这个思路非常有趣，并且与在 [word embedding](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html)中叙述过的关系很类似， [例如](https://developers.google.com/machine-learning/crash-course/embeddings/translating-to-a-lower-dimensional-space) ，distance(“king”, “queen”) ≈ distance(“man”, “woman”)。

设$$\phi_s$$和$$\phi_o$$分别为场景和目标的嵌入函数。模型通过最小化$$\phi_s(s_\text{pre}) - \phi_s(s_\text{post})$$和$$\phi_o(o)$$之间的距离（使用*n-pair*损失）来学习表示：
$$
\begin{aligned}
\mathcal{L}_\text{grasp2vec} &= \text{NPair}(\phi_s(s_\text{pre}) - \phi_s(s_\text{post}), \phi_o(o)) + \text{NPair}(\phi_o(o), \phi_s(s_\text{pre}) - \phi_s(s_\text{post})) \\
\text{where }\text{NPair}(a, p) &= \sum_{i<B} -\log\frac{\exp(a_i^\top p_j)}{\sum_{j<B, i\neq j}\exp(a_i^\top p_j)} + \lambda (\|a_i\|_2^2 + \|p_i\|_2^2)
\end{aligned}
$$
其中$$B$$表示一个批次的样本对（锚样本，正样本）。

当讲表示学习构造为度量学习时，[**n-pair loss**](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective)是一个常规选择。n-pairs损失将一个小批次中所有样本对的其他正样本都视为负的，而不是显式处理一组的三个样本（锚，正，负）。

嵌入函数$$\phi_o$$对于将目标$$g$$与一张图像一同表示来说性能很好。度量实际被抓取的目标$$o$$和目标的相近程度的reward函数定义为：$$r = \phi_o(g) \cdot \phi_o(o)$$。请注意，对于rewards的计算仅仅依赖于学习到的潜在空间，并不在于ground truth位置，因此它可用来在真实的机器人上进行训练。

![image-20210413214522895](/assets/img/typora-user-images/image-20210413214522895.png)

除过基于嵌入-相似度的reward函数，还有几个在grasp2vec框架中训练RL策略的tricks：

- *posthoc labeling P*：通过将随即抓取的目标标记为正确目标来增强数据集，如HER(Hindsight Experience Replay; [Andrychowicz, et al., 2017](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf))；
- *Auxiliary goal augmentation*：通过重标记带有未完成目标的transitions来进一步增强replay buffer；更准确地说，在每次迭代中，采样两个目标$$(g, g')$$，它们都用来向replay buffer中添加新的transitions。

相同场景的相同时步中的不同视角共享相同的embedding，即使在相同相机视角中嵌入可能随着时间变化(like in [FaceNet](https://arxiv.org/abs/1503.03832))。利用这个直觉  ，**TCN** (**Time-Contrastive Networks**; [Sermanet, et al. 2018](https://arxiv.org/abs/1704.06888)) 从多相机视角视频中学习。因此，嵌入捕捉了潜在状态而不是视觉相似度的语义含义。TCN嵌入通过 [triplet loss](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#triplet-loss)来训练。

训练数据通过同时拍摄相同场景的不同视角的视频来获得。所有视频都未被标记。

![image-20210413215410970](/assets/img/typora-user-images/image-20210413215410970.png)

TCN嵌入抽取对于相机配置具有不变性的视觉特征。它可被用来构建基于demo视频和潜在空间中的观测之间的欧氏距离的模仿学习的reward函数。

一个TCN的提升方法是，共同学习多个帧上的嵌入，而不是单个帧，便可得到**mfTCN** (**Multi-frame Time-Contrastive Networks**; [Dwibedi et al., 2019](https://arxiv.org/abs/1808.00928))。给定数个同步化后的相机视角的视频，$$v_1, v_2, \dots, v_k$$，$$t$$时刻的帧和之前的$$n-1$$从每个视频中以步长$$s$$选出的帧结合，映射至一个嵌入向量，得到一个尺寸为$$(n−1) \times s + 1$$的lookback窗口。每个帧首先通过一个CNN以提取低层级特征，然后我们使用3D时间卷积以在时序上结合帧。利用 [n-pairs loss](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#n-pair-loss)训练模型。

![image-20210413220129721](/assets/img/typora-user-images/image-20210413220129721.png)

训练数据通过以下步骤得到：

1. 首先我们构建两对视频片段。每对包含不同相机视角但是有着同步时步的两个视频片段。这两对片段在时序上应该离得很远；
2. 同时从相同对的每个视频片段采样固定数目的帧，采样步长相同；
3. 在n-pair损失中，相同时步的帧作为正样本来训练，其他的为负样本。

mfTCN嵌入可捕捉场景中的目标位置和速度（例如，在cartpole中），也可作为策略的输入。

### Autonomous Goal Generation

**RIG** (**Reinforcement learning with Imagined Goals**; [Nair et al., 2018](https://arxiv.org/abs/1807.04742))描述了一个利用非监督表示学习来训练一个以目标为条件goal-conditioned的策略的方法。策略通过首先想象“假”目标，然后尝试着去完成它，来从自监督实践中学习。

![image-20210413220959764](/assets/img/typora-user-images/image-20210413220959764.png)

此任务是控制一个机械臂，让它推动一个桌子上的小橡胶圆盘至理想位置。图中展示了这个理想位置，或者说目标。在训练过程中，它通过$$\beta$$-VAE encoder来学习状态$$s$$和目标$$g$$的潜在嵌入，控制策略在潜在空间中完全运作。

假设 [β-VAE](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#beta-vae)有一个encoder$$q_\phi$$，它将输入状态映射至潜在变量$$z$$（$$z$$通过高斯分布来建模），还有一个decoder $$p_\psi$$来将$$z$$映射回状态。将RIG中的encoder设定为β-VAE encoder的平均值。
$$
\begin{aligned}
z &\sim q_\phi(z \vert s) = \mathcal{N}(z; \mu_\phi(s), \sigma^2_\phi(s)) \\
\mathcal{L}_{\beta\text{-VAE}} &= - \mathbb{E}_{z \sim q_\phi(z \vert s)} [\log p_\psi (s \vert z)] + \beta D_\text{KL}(q_\phi(z \vert s) \| p_\psi(s)) \\
e(s) &\triangleq \mu_\phi(s)
\end{aligned}
$$
reward是状态嵌入向量和目标嵌入向量之间的欧氏距离：$$r(s, g) = -\|e(s) - e(g)\|$$。与 [grasp2vec](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#grasp2vec)相似，RIG通过潜在目标重标记，也应用数据增强操作：准确地说，一半目标都是从先前中随机生成的，剩下的一般通过HER进行选择。同样与grasp2vec相同，reward并不依赖于任何ground truth状态，而仅依赖于学习到的状态编码，因此它可被用来在真实的机器人上进行训练。

![image-20210414104550953](/assets/img/typora-user-images/image-20210414104550953.png)

随着RIG而来的问题是，在想象出的目标图片中缺乏目标变化。如果$$\beta$$-VAE仅仅利用一个黑色橡胶圆盘来训练，它就不能创建出带有其他物体的目标，如不同形状和颜色的物块。之后的一个改进将$$\beta$$-VAE替换为**CC-VAE** (Context-Conditioned VAE; [Nair, et al., 2019](https://arxiv.org/abs/1910.11670))，它的灵感来源于目标生成中的**CVAE** (Conditional VAE; [Sohn, Lee & Yan, 2015](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models))。

![image-20210414104938893](/assets/img/typora-user-images/image-20210414104938893.png)

CVAE依赖于上下文变量$$c$$。它训练encoder $$q_\phi(z \vert s, c)$$和decoder $$p_\psi (s \vert z, c)$$，请注意，$$c$$对于它们来说都是已知的。CVAE损失对于从输入状态$$s$$开始并经过信息瓶颈的信息进行处罚，但还涉及到从$$c$$开始至encoder和decoder的*无约束 unrestricted* 信息流。
$$
\mathcal{L}_\text{CVAE} = - \mathbb{E}_{z \sim q_\phi(z \vert s,c)} [\log p_\psi (s \vert z, c)] + \beta D_\text{KL}(q_\phi(z \vert s, c) \| p_\psi(s))
$$
为了创建合理的目标，CC-VAE依赖于一个起始状态$$s_0$$，因此生成的目标仍然与$$s_0$$中物体是相同类型。这种目标一致性是必须的；例如，如果当前场景包含一个红色橡胶圆圈，但是目标有一个蓝色物块，这就会让策略很迷惑。

除过状态encoder $$e(s) \triangleq \mu_\phi(s)$$，CC-VAE还训练第二个卷积enoder $$e_0(.)$$以将起始状态$$s_0$$转换为一个紧密上下文表示$$c = e_0(s_0)$$。这两个encoder特地被设计为不同的，并不共享权重，因为我们希望它们编码图象变化image variation的不同因子factor。除了CVAE的损失函数，CC-VAE还添加了一个额外项，以学习将$$c$$重构建回$$s_0$$，$$\hat{s}_0 = d_0(c)$$。
$$
\mathcal{L}_\text{CC-VAE} = \mathcal{L}_\text{CVAE} + \log p(s_0\vert c)
$$
![image-20210414111056330](/assets/img/typora-user-images/image-20210414111056330.png)

### Bisimulation

任务无偏表示Task-agnostic representation（例如，一个意欲表示系统中所有动态的模型）可能使得RL算法有所偏离，因为不相关的信息也被表示了。比如，如果我们只想训练一个自动encoder以重构建输入图像，则无法保证全部被学习到的表示对于RL来说是有用的。因此，我们需要从基于重构建的表示学习出发，走得更远——如果我们仅仅想学习对于模型相关的信息的话，因为不相关的细节对于重构建来说也很重要。

基于bisimulation的控制表示学习并不依赖于重构建，但它的目的是基于状态在MDP中的行为相似度来为状态分类。

**Bisimulation** ([Givan et al. 2003](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.2493&rep=rep1&type=pdf))指的是两个有着相似长期行为的状态之间的等价关系。*Bisimulation metrics*可度量此种关系，因此我们便可将状态结合以将高维状态空间压缩进一个更小的空间，以促进计算效率。两个状态间的*Bisimulation distance*与这两个状态的行为不同的程度相一致。

给定一个 [MDP](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#markov-decision-processes) $$ \mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$ 和一个 bisimulation关系 $$B$$，在关系 $$B$$（例如，$$s_i B s_j$$）下相等的两个状态应对所有动作具有相同的中间reward，并且在所有下一个bisimilar状态上都具有相同的转移概率：
$$
\begin{aligned}
\mathcal{R}(s_i, a) &= \mathcal{R}(s_j, a) \; \forall a \in \mathcal{A} \\
\mathcal{P}(G \vert s_i, a) &= \mathcal{P}(G \vert s_j, a) \; \forall a \in \mathcal{A} \; \forall G \in \mathcal{S}_B
\end{aligned}
$$
其中$$\mathcal{S}_B$$是关系$$B$$下的部分状态空间。

请注意，$$=$$ 始终是一个bisimulation关系。最有趣的bisimulation关系是maximal bisimulation关系 $$\sim$$，它定义了一个拥有 *最少* 状态组的部分空间$$\mathcal{S}_\sim$$。

![image-20210414134354545](/assets/img/typora-user-images/image-20210414134354545.png)

**DeepMDP** ([Gelada, et al. 2019](https://arxiv.org/abs/1906.02736)) 拥有与bisimulation度量相似的目标，它简化了RL任务中的高维观测，并且通过最小化两个损失来学习潜在空间：

1. rewards的预测以及
2. 所有下一个潜在状态的分布预测。

$$
\begin{aligned}
\mathcal{L}_{\bar{\mathcal{R}}}(s, a) = \vert \mathcal{R}(s, a) - \bar{\mathcal{R}}(\phi(s), a) \vert \\
\mathcal{L}_{\bar{\mathcal{P}}}(s, a) = D(\phi \mathcal{P}(s, a), \bar{\mathcal{P}}(. \vert \phi(s), a))
\end{aligned}
$$

其中$$\phi(s)$$是状态$$s$$的嵌入；所有带有bar的符号都是在同一个MDP中、但在潜在低维观测空间中运行的函数（reward函数$$R$$和转移函数$$P$$）。在此，嵌入表示$$\phi$$可与bisimulation度量相连接，因为bisimulation距离在潜在空间中的上限为L2距离。

函数$$D$$度量了两个概率分布之间的距离，应该慎重选择此函数。DeepMDP主要关注*Wasserstein-1*度量（也被称作[“earth-mover distance”](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html#what-is-wasserstein-distance)）。在度量空间$$(M, d)$$（例如，$$d: M \times M \to \mathbb{R}$$）中，分布$$P$$和$$Q$$之间的Wasserstein-1距离为：
$$
W_d (P, Q) = \inf_{\lambda \in \Pi(P, Q)} \int_{M \times M} d(x, y) \lambda(x, y) \; \mathrm{d}x \mathrm{d}y
$$
其中$$\Pi(P, Q)$$是$$P$$ 和$$Q$$ 的所有[couplings](https://en.wikipedia.org/wiki/Coupling_(probability))集合，$$d(x, y)$$定义了将一个particle从点$$x$$移至点$$y$$的成本。

根据Monge-Kantorovich对偶性，Wasserstein度量拥有一个对偶形式：
$$
W_d (P, Q) = \sup_{f \in \mathcal{F}_d} \vert \mathbb{E}_{x \sim P} f(x) - \mathbb{E}_{y \sim Q} f(y) \vert
$$
其中$$\mathcal{F}_d$$是度量$$d$$ 下的1-Lipschitz函数的集合：$$\mathcal{F}_d = \{ f: \vert f(x) - f(y) \vert \leq d(x, y) \}$$。

DeepMDP将此模型推广至Norm Maximum Mean Discrepancy (Norm-[MMD](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions))度量，以增进它的深度值函数界限的tightness，同时节省计算成本（Wasserstein的计算十分昂贵）。在实验中，他们发现转移预测的模型结构能够对性能造成很大影响。在训练model-free RL智能体时，添加DeepMDP损失作为辅助损失能够在大多数Atari游戏中显著改善性能。

**Deep Bisimulatioin for Control** (short for **DBC**; [Zhang et al. 2020](https://arxiv.org/abs/2006.10742)) 学习那些对于RL任务中的控制有很大帮助的观测的潜在表示，其中不带任何domain knowledge或像素级重构建。

![image-20210414144336445](/assets/img/typora-user-images/image-20210414144336445.png)

与DeepMDP相似，DBC通过学习reward模型和转移模型来建模动态量。这两个模型都在潜在空间$$\phi(s)$$中运行。对于嵌入$$\phi$$的优化依赖于 [Ferns, et al. 2004](https://arxiv.org/abs/1207.4114) (Theorem 4.5)和[Ferns, et al 2011](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.295.2114&rep=rep1&type=pdf) (Theorem 2.6)中的一个重要结论。

> 给定一个折扣因子$$c \in (0, 1)$$，策略$$\pi$$被连续优化，$$M$$是状态空间$$S$$上的bounded [pseudometric](https://mathworld.wolfram.com/Pseudometric.html)空间，我们定义$$\mathcal{F}: M \mapsto M$$：
> $$
> \mathcal{F}(d; \pi)(s_i, s_j) = (1-c) \vert \mathcal{R}_{s_i}^\pi - \mathcal{R}_{s_j}^\pi \vert + c W_d (\mathcal{P}_{s_i}^\pi, \mathcal{P}_{s_j}^\pi)
> $$
> 则$$\mathcal{F}$$有一个独一无二的固定点$$\tilde{d}$$，它是$$\pi^*$$-bisimulation度量，并且$$\tilde{d}(s_i, s_j) = 0 \iff s_i \sim s_j$$。

给定数个观测对批次和对于$$\phi$$的训练损失$$J(\phi)$$，最小化on-policy bisimulation度量和潜在空间中欧氏距离的均方差：
$$
J(\phi) = \Big( \|\phi(s_i) - \phi(s_j)\|_1 - \vert \hat{\mathcal{R}}(\bar{\phi}(s_i)) - \hat{\mathcal{R}}(\bar{\phi}(s_j)) \vert - \gamma W_2(\hat{\mathcal{P}}(\cdot \vert \bar{\phi}(s_i), \bar{\pi}(\bar{\phi}(s_i))), \hat{\mathcal{P}}(\cdot \vert \bar{\phi}(s_j), \bar{\pi}(\bar{\phi}(s_j)))) \Big)^2
$$
其中$$\bar{\phi}(s)$$表示带有停止梯度stop gradient的$$\phi(s)$$，$$\bar{\pi}$$是平均策略输出。学习到的reward模型$$\hat{\mathcal{R}}$$是deterministic的，学习到的前向动态模型$$\hat{\mathcal{P}}$$输出一个高斯分布。

DBC的基础是SAC，但是它在潜在空间中运行：

![image-20210414145417629](/assets/img/typora-user-images/image-20210414145417629.png)
