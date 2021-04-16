---
title: On-Policy & Off-Policy in PG
layout: post
categories: reinforcement-learning
tags: reinforcement-learning
date: 2021-04-13 15:50
excerpt: On-Policy & Off-Policy in PG
---

# On-Policy & Off-Policy in PG

## On-Policy

进行更新迭代的那个策略与从中探索并获取样本的策略是同一个策略，获取的样本用于计算PG。

因为每次更新都需要重新用更新过的策略以获取样本，用新样本来计算PG，而旧样本不能重复使用，因此**sample efficiency**很低，策略更新的效率也很低。

## Importance Sampling

### Distribution Ratio

其本质是，轨迹的reward可通过由一个不同的策略/分布而来的样本进行计算，之后再用轨迹的**distribution ratio**来重新校准。原本的优化目标函数为：
$$
J(\theta)=\mathbb{E}_{\tau\sim \pi(\tau)}[r(\tau)]
$$
应用Importance Sampling后：
$$
J(\theta)=\mathbb{E}_{\tau\sim \tilde{\pi}(\tau)}[\frac{\pi_\theta(\tau)}{\tilde{\pi}(\tau)}r(\tau)]
$$
由于$\pi_\theta(\tau)=p(s_1)\prod^{T}_{t=1}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$，可见转移概率独立于策略函数，则上式中的distribution ratio为：
$$
\frac{\pi_\theta(\tau)}{\tilde{\pi}(\tau)}=\frac{\prod^T_{t=1}\pi_\theta(a_t|s_t)}{\prod^T_{t=1}\tilde{\pi}(a_t|s_t)}
$$
由此，轨迹样本便可从更新前（或历史的）策略中获得，这样无论策略是否被更新，我们都可以更新replay memory，将importance sampling用于PG。

求目标函数的梯度，有：
$$
\nabla_\theta J(\theta)=E_{\tau\sim \tilde{\pi}_\theta(\tau)}
\left[\sum^T_{t=1}\nabla_\theta log\pi_\theta(a_t|s_t)
\left(\prod^T_{t'=1}\frac{\pi_\theta(a_t'|s_t')}{\tilde{\pi}_\theta(a_t'|s_t')}
\left(\sum^{T}_{t'=t}r(a_t',s_t')
\right)\right)\right]
$$

### Dimensionality Curse

上式的目标函数梯度的表达形式是相乘的形式，可能导致dimensionality curse，因此我们将目标函数写为：
$$
J(\theta)=
\sum^T_{t=1}E_{s_t,a_t\sim p_\theta(s_t,a_t)}[r(s_t,a_t)] \\ 
\qquad \qquad \ =\sum^T_{t=1}E_{s_t\sim p_\theta(s_t)}\left[ 
E_{a_t\sim \pi_\theta(a_t,s_t)r(s_t,a_t)}
\right]
$$
将importance sampling应用于上式，得到：
$$
J(\theta')=
\sum^T_{t=1}E_{s_t\sim p_\theta(s_t)}
\left[
\frac{p_{\theta'(s_t)}}{p_{\theta(s_t)}}
E_{a_t\sim \pi_\theta(a_t|s_t)}
\left[
\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}
r(s_t,a_t)
\right]
\right]
$$
在此轨迹样本来源于$\pi_\theta$，转移概率为$p_\theta$，需要更新优化的策略为$\pi_{\theta'}$，转移概率为$p_{\theta'}$。

如果我们能够对$\pi$更新的程度（也就是说策略距离当前策略来说走得有多远）作出限制，则可略去$\frac{p_{\theta'(s_t)}}{p_{\theta(s_t)}}$这一项，因为这两个相似策略间的状态概率分布应该也近似相同，则我们可将目标函数写为：
$$
\mathop{max\ mize}_\limits{\theta}\ 
\hat{\mathbb{E}}_t
\left[
\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
\hat{A}_t
\right] \\
\qquad \qquad \qquad \; \; \; \ \ 
s.t. \quad
\hat{\mathbb{E}}_t[KL[\pi_{\theta_{old}}(.|s_t),\ 
\pi_{\theta}(.|s_t)]] \leq
\delta
$$
这里的**constraint**指的是，我们限制更新后策略和更新前策略之前的差异不能大于$\delta$，即限制$\pi$变化的程度。以此constraint迭代目标策略，获得optimal policy $\pi^*$。

### Why We Use Importance Sampling?

深度学习中的传统优化方法比如gradient descent，它的假设前提是输入数据的分布具有相对恒常的性质。但RL tasks显然不具有这种性质，这使得RL中learning rate的调整变得非常困难。假如更新步的太小，则convergence很慢，假如更新步太大，则有做出坏action的可能性，这使得下一episode开始时，迭代变得很困难。

Importance Sampling是TRPO，PPO的基础。加上上述的constraint可以让我们知道对于$\pi$的更新程度最大能够多大，超过这个最大限制就会让$\pi$变得inconvincible。因此，**这个置信区间使我们不至于过度优化。**
