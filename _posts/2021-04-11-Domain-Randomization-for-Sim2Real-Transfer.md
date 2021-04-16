---
title: Domain Randomization for Sim2Real Transfer
layout: post
categories: reinforcement-learning
tags: reinforcement-learning meta-learning robotics
date: 2021-04-11 12:00
excerpt: Domain Randomization for Sim2Real Transfer
---

# Domain Randomization for Sim2Real Transfer

> 如果一个模型或策略很大程度上是在模拟器中训练但是我们却期望它在实际的机器人上运行，那么它肯定会面临一个sim2real的断层问题。Domain Randomization DR是一种简单但有效的可弥合此种断层的方法，它通过随机化训练环境的性质（参数）来实现。

在机器人学中，一个最难的问题就是如何将模型迁移至真实世界中。由于DRL算法的样本低效率并且在实际的机器人上的数据收集的成本太高，我们常常在模拟期中训练模型，此种环境理论上会提供无限量的数据。然而你，这种在模拟器和物理世界间的真实断层经常会导致使用实际的机器人时的失败。此种断层可被物理参数间的非一致性所引发，并且更为致命的是，不正确的物理建模。

为了弥合这种sim2real断层，我们需要改善模拟器并使其更加接近于现实环境。有如下方法：

- System Identification

System Identification会为物理系统来建立一个数学性质的模型。在RL的语境下，数学模型就是模拟器。为了使得模拟器更加具有真实性，小心谨慎的测定是非常必要的。

然而不幸的是，完备的测定是非常昂贵的。并且，一个相同机器上的许多物理参数可能会随着温度、湿度、位置或它的wear-and-tear发生非常大的变化。

- Domain Adaption

Domain Adaption DA指的是迁移学习技术的集合，这种技术可通过由task模型引发的映射或正则化来更新在sim中的数据分布以匹配real。

许多DA模型，尤其是针对于图像分类或端到端的基于图像的RL任务，都是建立在adversarial loss或者[GAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)之上的。

- Domain Randomization

我们能够通过利用Domain Randomization DR来创建一个拥有随机性质的模拟环境的集合，并且训练一个可在所有这些环境之上运行的模型。

与此类模型可应用于真实世界环境相似的是，实际系统应该是一个有着既丰富训练变量分布的样本。

DA与DR都是非监督的。与需要相当真实数据样本以获得分布的DA相比，DR可能只需要一点点或不需要真实数据。DR即为本篇的重点。

![img](/assets/img/typora-user-images/031cbe76-4bac-4231-8f6e-b77a624f3979.png)

{:.table-of-content}
* TOC
{:toc}

## What is Domain Randomization?

为了使得定义更加统括，将那个我们了解其全部的环境叫做**source domain**，将那个我们想把模型迁移至其中的环境叫做**target domain**。我们在source domain中进行训练。我们可控制配置为$$\xi$$ 的source domain$$e_\xi $$中的随机化参数集$$N$$ ，其配置从一随机化空间中采样，$$\xi\in \Xi \subset \mathbb{R}^N$$ 。

在策略训练过程中，episodes从带有随机化的source domain中采集。因此，此策略将会运行在一系列环境当中，并且学会泛化。训练策略参数$$\theta $$以最大化期望回报$$R(.)$$在一个配置的分布上的在平均值：

$$
\theta^*=argmax_\theta \mathbb{E}_{\xi \sim \Xi}[\mathbb{E}_{\pi_\theta,\tau \sim e_\xi}[R(\tau)]]
$$
其中$$\tau_\xi$$ 是在一个根据$$\xi $$来随机化的source domain中收集到的轨迹。根据这种方式，*“source和 target domains间的差异就可以根据在source domain中的变异性来建模。”*[Peng et al. 2018](https://arxiv.org/abs/1710.06537)

## Uniform Domain Randomization

DR的最初形式[Tobin et al, 2017](https://arxiv.org/abs/1703.06907)[Sadeghi et al. 2016](https://arxiv.org/pdf/1611.04201.pdf)中，每个随机化参数$$\xi_i$$ 由一个区间来限制，$$\xi\in[\xi_i^{low},\xi_i^{high}] ，i=1,...,N $$每个参数都是在此区间内均匀采样的。

这些随机化参数可控制景象的外观，包括但不限于以下这些（如图2）.一个在模拟的、随机化的图象上训练的模型能够迁移至真实的、非随机化的图象上。

- 位置，形状，物体的颜色；
- 材料质地；
- 光照条件；
- 加至图象当中的随机噪声；
- 模拟器中相机的的位置，方向，观测位

![img](/assets/img/typora-user-images/b8f33590-7950-4732-870e-bf75cedc2e75.png)

模拟器中的物理动态也可被随机化[Peng et al. 2018](https://arxiv.org/abs/1710.06537)。有研究表明*recurrent策略*可适用于不同的物理动态中，包括部分可观测现实。物理动态特性包括但不限于：

- 物体的质量和维度；
- 机器人体的质量和维度；
- 关节的湿度、kp和摩擦；
- PID控制器的增益；
- 关节限度；
- 动作延迟；
- 观测噪声；

在OpenAI Robotics中，有了视觉和动力学DR，我们就可以学习一个可在真实的、灵活的机械臂[OpenAI, 2018](https://arxiv.org/abs/1808.00177)上运行的策略。我们的控制人物是教给机械臂去持续转动一个物体至50个接续的随机目标方向。此任务中的sim2real断层问题非常大，这是由机器人和物体间极多的同步接触以及物体碰撞和其他动作的并不完美的模拟所引起的。一开始，此策略几乎不能存活超过5秒钟，也就是不掉落此物体。但在DR的帮助下，策略可最终进化至能够在真实环境中运行的极好。

## Why Does Domain Randomization Work?

现在你可能会问了，为什么DR的表现如此之好？DR的思想真的非常简单。这有两个我觉得最令人信服的non-exclusive解释。

### DR as Optimization

有一种观点[Vuong, et al, 2019](https://arxiv.org/abs/1903.11774)是将DR中的learning随机化参数视为双层优化bilevel optimization。假设我们现在知道真实环境$$e_{real}$$ 并且随机化配置是从参数化分布$$\phi$$，$$\xi\sim P_{\phi}(\xi)$$ 中采样的，我们想学习一个可在$$e_{real}$$ 中获得最好表现的分布，策略$$\pi_{\theta} $$就是在此分布上进行训练的：

$$
\phi^*=argmin_\phi \mathcal{L}(\pi_{\theta^*(\phi)};e_{real}) 

where \theta^*(\phi)=argmin_\theta\mathbb{E}_{\xi \sim P_\phi(\xi)}[\mathcal(\pi_\theta;e_\xi)]
$$
其中$$\mathcal{L}(\pi;e)$$ 是策略$$\pi$$ 在环境$$e $$中进行评估的损失函数。

虽然随机化范围是从均匀DR中手动选择的，但它常常包含domain knowledge和基于迁移性能的数轮trial-and-error调整。本质上，这是一个对于最优$$\mathcal{L}(\pi_{\theta^*(\phi)};e_{real}) $$的$$\phi $$的手动优化调整过程。

下一章节的Guided DR就是由本节观点展开的，目的是去做自动bilevel优化以及学习最好的参数分布。

### DR as Meta-Learning

在我们的learning dexterity项目[OpenAI, 2018](https://arxiv.org/abs/1808.00177)中，我们训练了一个LSTM策略以在不同的环境动态中泛化。我们观测到，只要机器人完成了第一次转动，那么它成功完成后续任务的时间就会大大缩短。并且，一个不具有memory的FF策略是不能够迁移至一个真实机器人的。这两样都是策略动态learning和adapting至一个新环境的证据。

不完全地，DR构成了许多不同种的任务。在RNN中的memory使得策略在真实世界设置中达到任务以及更进一步的任务的[meta-learning](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)。

## Guided Domain Randomization

Vanilla DR假设我们并不知道真实数据，因此随机化配置的采样与模拟环境中的广阔程度与均匀程度尽可能相同，以期望真实环境可以在这种很广阔的分布下被覆盖到。考虑一个更加成熟的策略这种想法是很合理的——将均匀采样替换为从任务performance，真实数据或模拟中获得的引导guidance。

我们使用guided DR的动机是为了通过避免在不真实环境下训练模型而节约计算资源。另一个动机是避免那些可能由于过度随机化分布而产生的不能实际产生作用的方案，它们可能妨碍成功的policy learning。

### Optimization for Task Performance

假设我们训练了一个带有不同随机化参数$$\xi \sim P_{\phi}(\xi) $$的策略群，其中$$P_\xi $$是参数为$$\phi$$ 的关于$$\xi $$的分布。然后我们决定在target domain中的downstream task上尝试每一个策略，以获得反馈。这种反馈告诉我们配置$$\xi $$有多么好，以及提供优化$$\phi$$ 的信号。

AutoAugment[Cubuk, et al. 2018](https://arxiv.org/abs/1805.09501)的灵感来源于[NAS](https://ai.google/research/pubs/pub45826)，它将图像分类的学习最好数据提升操作的问题构造成了一个RL问题。请注意，AutoAugment不是为了sim2real迁移而被提出的，但是通过task performance进入到了DR guided的分类当中。单个的augmentation configuration在evaluation set上进行测试，性能的改善被用作reward以训练一个PPO policy。此策略输出对于不同数据集的不同augmentation strategies；比如，对于CIFAR-10，AutoAugment主要挑选出基于颜色的变换，然而对于ImageNet它更倾向于基于几何的。

[Ruiz (2019) ](https://arxiv.org/abs/1810.02513)研究了在RL问题中的将task feedback视作reward的问题，并提出了一个基于RL的方法，叫做“learning to simulate”，以调整\xi 。使用主要任务的验证集上的性能度量以训练一个策略，这个策略将预测$$\xi$$ ，并被建模为多变量高斯分布。总之，这个idea与AutoAugment很相似，都将NAS应用于data generation上。根据他们所做的实验，即使主要任务模型是不收敛的，它仍可以为data generation策略提供一个合理的信号。

![img](/assets/img/typora-user-images/d4725d48-a8e6-4c53-985a-7854235592d3.png)

进化算法是另一种可行的方法，其中feedback被视为guiding 进化的fitness[Yu et al, 2019](https://openreview.net/forum?id=H1g6osRcFQ)。在这项研究中，当fitness是target环境中的$$\xi$$-conditional 策略的性能时，他们使用[CMA-ES](https://en.wikipedia.org/wiki/CMA-ES)（协方差矩阵适应进化策略covariance matrix adaption evolution strategy）。在附录当中，他们将CMA-ES与其他建模$$\xi $$动态的方法进行了比较，包括贝叶斯优化或者神经网络。主要的论点是，这些方法都不如CMA-ES稳定，并且样本效率也不如。有趣的是，当把$$P(\xi) $$建模为神经网络时，LSTM的表现远远超过了FF。

一些人认为，sim2real gap是appearance gap与content gap的结合；例如，大多数灵感来源于GAN的DA模型都关注appearance gap。Meta-Sim[Kar, et al. 2019](https://arxiv.org/abs/1904.11621)的主要目的是通过产生特定于任务的、合成的synthetic数据集来缩小content gap。在这种情况下，synthetic scenes是通过具有不同性质的对象的层级以及对象间的关系来参数化的（例如，位置，颜色等）这种层级由一个类似于结构DR(SDR[Prakash et al., 2018](https://arxiv.org/abs/1810.10093))的概率情景语法probabilistic scene grammar来描述的，并且我们假设它之前就是已知的。训练模型$$G $$以提升情景性质分布scene properties $$s$$：

1. 首先学习先验知识：预训练$$G$$ 以学习恒等函数$$G(s)=s$$ ;
2. 最小化真实数据分布和模拟数据分布间的MMD损失。它包括不可微分渲染中的反向传播。这篇论文通过扰动$$G(s) $$的属性来从数据上计算损失；
3. 在合成数据上进行训练时，最小化REINFORCE任务损失，但在真实数据上进行评估。

不幸的是，此方法族并不适合sim2real。无论是RL策略还是EA模型都需要大量真是样本。将真实机器人上的实时反馈集加入到训练回环中是非常昂贵的。计算资源和实时数据集的trade-off取决于任务类型。

### Match Real Data Distribution

理统真实数据以引导DR很大程度上类似于进行系统识别或DA。DA背后的核心观点就是改善合成数据以匹配真实数据分布。在真实数据引导的DR中，我们希望学习随机化参数$$\xi$$ ，它使得模拟器中的状态分布接近真实世界中的状态分布。

SimOpt[Chebotar et al, 2019](https://arxiv.org/abs/1810.05687)首先在一个初始随机化分布$$P_\phi(\xi)$$ 的基础上进行训练，得到策略$$\pi_{\theta,P_\phi} $$。然后，在模拟器和实际机器人上都部署应用此策略以分别收集轨迹$$\tau_\xi$$ 和$$\tau_{real}$$ 。优化目标是最小化sim轨迹和real轨迹之间的差异：

$$
\phi^*=argmin_\phi\mathbb{E}_{\xi\sim P_\phi(\xi)}[\mathbb{E}_{\pi_\theta, P_\phi(\xi)}[D(\tau_{sim},\tau_{real})]]
$$
其中$$D(.)$$ 是一个基于轨迹差异的度量。就像“Learning to simulate”论文，SimOpt也必须解决如何在不可微分模拟器中进行梯度传播的棘手问题。它使用了一个叫做[ relative entropy policy search](https://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewFile/1851/2264)的方法。

![img](/assets/img/typora-user-images/2f452a76-8f8e-489f-ba3e-07e92e7c7e4a.png)

RCAN[James et al., 2019](https://arxiv.org/abs/1812.07252)是Randomized-to-Canonical Adaption Networks的所辖，端到端RL任务的DA和DR的很好的结合。在sim中训练image-conditional GAN([cGAN](https://arxiv.org/abs/1611.07004))以将domain-randomized图象转换为非随机化版本（aka "canonical version"）。之后，利用一个相同的模型将真实图象转换为对应的模拟版本，则此智能体就可以在训练中遇到时使用相同的观测。潜在的假设仍然是domain-randomized sim图象的分布足够宽以覆盖真实世界样本。

![img](/assets/img/typora-user-images/a78221d5-11c2-4d5d-8b4e-08757b0db2ac.png)

为了实施基于视觉的机械臂抓取，我们在模拟器中对RL模型进行端到端的训练。在每一个时间步中都进行随机化，包括托盘divider的位置、抓取的对象、随机的质地，还包括灯光的位置、方向以及颜色。这种canonical version是默认模拟器外观。RCAN尝试学习一个生成器$$G$$：$$randomized image \rightarrow\{canonical\  image, segmentation,depth\}$$。

其中分割masks和深度图象被用来作为辅助任务。RCAN比起均匀DR来说具有更好的零次迁移zero-shot transfer，即使它们两个都比在真实图象上训练的模型要差。从概念上来说，RCAN在[GraspGAN](https://arxiv.org/abs/1709.07857)的反方向上操作，它可以将合成图像通过domain adaption转换为真实图象。

### Guided by Data in Simulator

网络驱动的DR也被称作DeceptionNet[Zakharov et al., 2019](https://arxiv.org/abs/1904.02750)，它的思想来源是，那些随机化实际上有效的学习以弥合图像分类任务的domain gap。

随机化在一系列具有encoder-decoder架构的欺骗模块deception modules中得以应用。欺骗模块是专为变换图像而设计的；比如替换背景、添加变形、改变光线，等等。其他识别网络则通过在变换后图象上实行分类以解决主任务。

训练过程包括两步：

1. 识别网络固定时，通过在反向传播过程中应用反响梯度来最大化预测标签与真实标签的差异。因此，欺骗模块便可以学习那些最令人迷惑的技巧。
2. 欺骗模块固定时，将输入图像改变以训练识别网络。

![img](/assets/img/typora-user-images/ed0bcdf9-8385-4e04-8a2e-b59d33b299b5.png)

训练欺骗网络时的反馈由downstream分类器提供。但并不是尝试去最大化任务性能，随机化模块的目的反而是去创建更加困难的情况。一个很大的缺点是，你需要为了不同数据集或不同任务去手动设计不同的欺骗模块，这使得测量变得很困难。考虑到它是零阶的，它的结果仍然比SOTA DA方法在MNIST和LineMOD上要差。

相似的是，活跃DR Active domain randomization ADR[Mehta et al., 2019](https://arxiv.org/abs/1904.04762)也依赖于sim数据以创建更困难的训练样本。ADR在给定的随机化范围中寻找最大信息量的环境变种，其中informativeness作为随机化和参考（初始的、非随机化的）环境范例中的策略迭代的差异来被测量。听起来有点像[SimOpt](https://lilianweng.github.io/lil-log/2019/05/05/domain-randomization.html#match-real-data-distribution)是不是？注意到SimOpt测量的是sim和real迭代中的差异，然而ADR测量的是随机化sim和非随机化sim的，避免了昂贵的真实数据收集。

![img](/assets/img/typora-user-images/b97f72d1-475e-4c89-857f-ab8bdeeb3215.png)

精确的训练过程为：

1. 给定一个策略，在参考和随机化环境中运行它，并分别收集两个轨迹集合；

2. 训练discriminator模型以判断一个rollout的轨迹除了参考run外是否被随机化。这个预测的

   $$log\ p$$ （被随机化的概率）将作为reward。不同的随机化和参考rollout越多，则预测越简单，reward就越高；

   - 直觉上，如果一个环境比较简单，则相同的策略就可以产生与参考环境中相似的轨迹。那么此模型就该通过鼓励做不同的行为来探索并reward。

3. discriminator的reward将提供给Stein Variational Policy Gradient SVPG particles[SVPG](https://arxiv.org/abs/1704.02399)，输出一个内含不同的随机化配置的集合。
