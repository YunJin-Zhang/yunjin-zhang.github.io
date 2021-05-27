---
title: Deep Reinforcement Learning for Autonomous Driving A Survey
layout: post
categories: reinforcement-learning
tags: reinforcement-learning autonomous-driving
date: 2021-05-26 15:00
excerpt: Deep Reinforcement Learning for Autonomous Driving A Survey
---

# Deep Reinforcement Learning for Autonomous Driving: A Survey   

## Abstract

- DRL算法；
- DRL算法应用的自动驾驶任务；
- 自动驾驶在现实世界应用的挑战；
- 相关领域如动作克隆、模仿学习、IRL；
- RL中模拟器、方法、稳定性验证的角色。

## Introduction

AD系统有着多种感知层级任务，以及多种任务（传统监督学习方法不再适用）。1. 当智能体的动作预测改变时，传感器的观测也改变；2. 监督性信号如碰撞时间TTC，侧面误差表示着智能体的动态量以及环境的不确定性，这就需要定义最大化的随机损失函数；3. 智能体需要学习环境中的新状况以及在驾驶中预测每一瞬间的最优决策。**我们的目标是解决一个序列决策过程。**

## AD系统里的成分

AD系统的pipeline：

![image-20210525215648181](/assets/img/typora-user-images/image-20210525215648181.png)

**传感结构**包括：多相机系列、雷达、LIDARs、GPS-GNSS系统（测量绝对位置）和惯性测量单元IMUs（提供车的3D位姿）。**感知模块**的目标是创建环境状态（包括道路位置、可驾驶区域、智能体位置等）的中间层级表示，感知模块中的不确定性会传播至信息链中的其他部分。**稳定感知**非常重要，因此冗余的信息源可提高检测的可信度。因此**检测任务**由以下任务结合：语义分割、运动估计、深度估计、地面检测等。

### 场景理解

此模块将从感知模块中获得的感知状态的中间层表示映射到高层级动作或决策模块，它可融合传感器提供的信息员，提供更抽象的场景知识。**信息融合**为决策过程提供更为统括和简化的环境的表示知识，并建模了传感器噪声以及检测到的不确定性信息。

### 定位和映射

一个场景被映射后，智能体的当前位置就可在地图中被定位到。传统映射技术通过语义目标检测被增强，并且定位后的**高精度地图**（HD maps）可被用作目标检测的先验知识。

### 规划和驾驶策略

轨迹规划是AD系统中的重要模块。根据HD地图或基于GPS地图的路线级别规划，此模块需要生成**运动级别**的命令以控制智能体。如果一个智能体的可控制自由度少于其全体**自由度**（**DOF**），则说此智能体是**非完整的（non-holonomic）**。**快速规划随机树（RRT）**算法是非完整算法，可通过随机抽样和无障碍路径生成来探索环境。RRT有许多辩题。

### 控制器

控制器给出路径上所需每点的速度、控制角度以及刹车动作。**轨迹追踪**则包含智能体每时刻的动态量的时序模型。**当前智能体控制方法**都基于**经典最优控制理论**。控制输入都在有限时间域中定义，并被限制在可控状态空间内。**速度控制**都基于传统闭环控制方法如PID、**模型预测控制MPC**等。MPC算法族的目标是当智能体追踪特定路径时，稳定其行为。最优控制和RL方法有其内在联系，最优控制可被视为智能体和环境动态量都被建模为可微方程的model-based强化学习方法。RL方法用以解决随机控制问题。

## 强化学习

## 强化学习拓展

此处讨论的扩展方法用以对RL算法在复杂问题领域中的可应用性、可扩展性、学习速度、收敛性能进行详述。

### Reward Shaping

由于奖励很稀疏或延迟，学习和训练可能很困难。可通过加入额外的知识来改善学习速度和收敛性能，这就是reward shaping。**奖励塑造**可通过塑造一个奖励方程来为合适的行为提供更频繁的反馈信号，这在稀疏奖励的情况下尤其有用。$$r'=r+f$$，$$f$$是由**塑造方程**$$F$$而来的额外回报。但加入奖励塑造洗漱可能带来这样的问题：对于增强后奖励函数最优的策略对于初始奖励函数来说并不最优。**Difference rewards（D）**和**potential-based reward shaping（PBRS）**是两种常用塑造方法。

### MARL

在MA系统里，常用描述为**随机游戏（SG）**，SG用元组$$<S,A_{1\dots N},T,R_{1\dots N}>$$来描述，$$N$$为智能体数量，$$S$$为系统状态集合，$$A_i$$是智能体$$i$$的动作集合，$$T$$为转移函数，$$R_i$$为智能体$$i$$的回报函数。当$$N=1$$时，SG就变成了MDP。系统的下一状态和每个智能体收到的回报取决于所有智能体的动作。每个智能体都有其局部状态描述$$s_i$$和回报函数$$R_i$$。在SG中，智能体可能有相同的目标（**协作式SG**），或者完全相反的目标（**竞争式SG**），这取决于特定应用的回报函数设定方式。

### MORL

多目标RL最大的区别在于回报函数，回报被定义为一个向量$$\mathbf{r}$$，其中包含每个独立的目标的回报，MORL多用于处理需要取得冲突或竞争的多目标方程的序列决策问题。MORL问题使用MDP或SG来描述，可将MDP或SG通过改变回报函数来拓展至**MOMDP**或**MOSG**。MORL算法寻求学习或近似**非主导方法**集合，并且使用**Pareto dominance**的概念来评估MORL方法。

### 状态表征学习SRL

SRL指的是用**特征提取**和**维度减少**方法来表示状态空间。SRL最简单的形式将高维向量$$o_t$$映射到一个小维度的隐藏空间$$s_t$$中，然后智能体学习从隐藏空间中映射到动作。训练SRL链是无监督学习，SRL可以是简单的**自编码器AE**，**VAE**，**GAN**以及预测下一状态的前向模型或是预测动作的的反向模型。一个好的状态表征应该是马尔可夫的。

### 从演示中学习Learning from Demonstrations（LfD）

LfD中，智能体从演示中学习如何执行任务，演示的形式通常是不带有反馈回报的状态动作对。LfD对于回报很稀疏的或输入域过大的初始探索是很重要的。单纯的LfD可用来初始化一个好的或安全的策略，则RL就可以通过与环境的交互来进化到一个更好的策略。将RL与LfD结合的方法有AlphaGo。给定初始演示，则无需显式的探索也可以获得近最优性能。**DQfD**可预训练智能体，通过将带有额外优先权的演示加入回放池的方法利用专家演示。**行为克隆BC**可作为监督学习方法基于演示将状态映射到行为。**IRL**从演示中推理回报方程，但成本过高。**生成对抗模仿学习GAIL**可避免IRL带来的内部循环。

## 自动驾驶任务中的强化学习

自动驾驶任务中，**强化学习应用的领域**包括：控制器优化、路径规划、轨迹优化、运动规划&动态路径规划、复杂导航任务的高级驾驶策略开发、高速路&交叉路口&合并道路&岔路口的基于场景的策略学习、从交通要素的目的预测的专家数据中进行学习的回报学习及IRL（以学习到确保安全并能够估计风险的策略）。

**自动驾驶任务**的**标准**包括：到达目的地走过的距离、自车的速度、保持自车处于静止状态、与其他道路用户或场景中物体的碰撞、碰到人行道、保持在小路上行驶、在躲避极限加速&刹车&驾驶时保持舒适和稳定状态、遵守交通规则。

### 状态空间、动作空间和回报

常规的自动驾驶任务的**状态空间特征**包括：自车的位置、朝向、速度以及自车传感器视野中的障碍。为了避免状态空间维度中的变化，经常使用**自车周围的Cartesian或Polar占位栅格图**，它还被**道路信息**所增强，如道路数量、路径曲率、自车的过去和未来的轨迹、Time to collision、场景信息（如交通规则和信号灯）。使用**原始传感器数据**可带来更精细的场景信息，但使用**压缩或抽象数据**可减少状态空间的复杂度，中间级表征有如**2D bird eye view(BEV)**。这种中间级表示保留了道路空间信息，而**基于图像的表示**不保留。

自动驾驶策略必须控制许多不同的**传动装置**。自动驾驶中的**连续值**传动结构包括控制角度、刹车和离合，其他的为**离散值**传动结构。为了减少复杂度并使用离散DRL算法可进行**离散化**，或者**log离散化**，但这样会带来一些问题。另一个选择是为连续值传动装置使用**policy-based算法**。**时序抽象选择框架temporal abstractions options framework**是一个简化选择动作的过程的方案，其中智能体选择options而不是低层级的动作，这些options表示可在多个时间步上扩展原始动作的子策略。

![image-20210526162342487](/assets/img/typora-user-images/image-20210526162342487.png)

### 运动规划&轨迹最优化

在动态环境和变化的车的动态量中的路径规划是一个很关键的问题。最近研究者提出了一种使用**全尺寸自动车辆**的DDPG的应用，首先在模拟环境中训练，然后成功的完成了现实世界中的实验。路径规划中的算法还有model-based DRL算法。RL对于控制也很适合，**经典RL方法**被用来在随机环境中执行最优控制任务，如线性环境中的**线性四倍调节器LQR**以及在非线性环境中的**迭代式LQR iLQR**，同时策略网络中进行参数的随机搜索也可达到与LQR一样的性能。

### 模拟器&场景生成工具

![image-20210526165241240](/assets/img/typora-user-images/image-20210526165241240.png)

模拟器用于训练并验证RL算法，上表概括了可模拟相机、LiDARs和雷达的高逼真度感知模拟器。一些模拟器还可提供车的状态和动态量。文章F. Rosique, P. J. Navarro, C. Fernández, and A. Padilla, **“A systematic review of perception system and simulators for autonomous vehicles research,”** Sensors, vol. 19, no. 3, p. 648, 2019. 10  列举了自动驾驶领域的传感器和模拟器。 **多种逼真度RL框架（MFRL）**提供多种模拟器，可在一系列模拟器中训练并验证RL算法，找到真实世界的近最优策略。基于Carlo模拟器的AD比赛**CARLA Challenge**拥有多种场景：自车失去控制、自车对未见过的障碍做处反应、道路变化。

### AD中的LfD和IRL

- Alvinn: An autonomous land vehicle in a neural network  
- Efficient training of artificial neural networks for autonomous navigation  
- End to end learning for self-driving cars  
- Explaining how a deep neural network trained with end-to-end learning steers a car  
- Learning driving styles for autonomous vehicles from demonstration  
- Learning to drive using inverse reinforcement learning and deep q-networks  

## 真实世界中的挑战&未来愿景

### 验证RL系统

不同的code-based、超参数值、top-k轮数都会对算法性能和泛化能力带来影响。同时，在模仿学习中向训练数据中加入复杂训练场景可以增加安全性。

Generating adversarial driving scenarios in high-fidelity simulators  

### 弥合sim2real的鸿沟

### 样本效率

**reward shaping**：可通过设计更为频繁的回报函数来让智能体学习中间目标，可使得通过更少的样本来学习得更快。

**IL boostrapped RL**：可通过模仿学习在线下学习到初始策略，然后再通过与环境交互来自我改进。

**Actor-Critic**：与经验回放结合的ACER，与TRPO结合可提高样本效率。

**迁移学习**：是提高样本效率的另一种方法，通过复用先前训练的策略来初始化目标任务的学习过程，可用更少的样本加快新策略的学习过程。

**Meta-Learning**：可使得智能体通过其先验知识快速适应新任务，并快速从很少的经验/交互中学习到新技巧。

**高效状态表征**

### 模仿学习中的探索问题

模仿学习中，演示中的状态分布并不包含所有状态，并且模仿学习假设动作都是i.i.d.的。一个方法是**Data Aggregation(DAgger)**，可从参考和训练测的探索中迭代收集训练样本。其他方法还有**Search-based Structure Prediction(SEARN)**，**Stochastic Mixing Iterative Learning(SMILE)**，**Chauffeurnet**等。

### 本质回报函数

回报函数可通过IRL方法获得。但在没有显式回报塑造和专家演示的任务中，智能体可使用**本质回报**或**本质动机**来评估动作的好坏。

### DRL中的复合安全性

对于基于模仿学习的系统，**Safe DAgger **引入了安全性策略，以预测主策略做出的决策的误差。另一种额安全策略将部分观测和主策略作为输入，返回一个二元标签以表明主策略是否偏离参考策略。同时也有针对于MARL的安全性策略方法，也有将DDPG与基于安全性的控制结合起来的方法。**可以证明，DRL与基于安全性的控制的结合法在大多数场景中表现良好。** **SORL**可使得DRL脱离局部最优、加速训练并避免危险状况，它对回报函数并不敏感。

### MARL视角下的AD

- Safe, multiagent, reinforcement learning for autonomous driving  
- Multi-agent connected autonomous driving using deep reinforcement learning  
- Deep multi agent reinforcement learning for autonomous driving
- Failure-scenario maker for rule-based agent using multiagent adversarial reinforcement  learning and its application to autonomous driving  
- Distributed multiagent coordinated learning for autonomous driving in highways based on dynamic coordination graphs  

![image-20210526220639141](/assets/img/typora-user-images/image-20210526220639141.png)

![image-20210526220658619](/assets/img/typora-user-images/image-20210526220658619.png)

![image-20210526220709462](/assets/img/typora-user-images/image-20210526220709462.png)


