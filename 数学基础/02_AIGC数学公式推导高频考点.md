# 目录

## 第一章 基础求导与损失函数推导

[1. Softmax + Cross Entropy 的梯度如何推导？](#q-001)
[2. Sigmoid + Binary Cross Entropy 的梯度如何推导？](#q-002)
[3. MSE 为什么对应高斯噪声假设？](#q-003)
[4. MAE 为什么对应 Laplace 噪声假设？](#q-004)
[5. LayerNorm 和 RMSNorm 的核心反向传播直觉是什么？](#q-005)

## 第二章 概率统计与信息论推导

[6. 最大似然估计为什么等价于最小化负对数似然？](#q-006)
[7. Cross Entropy、KL 散度和最大似然之间有什么关系？](#q-007)
[8. KL 散度非负性如何证明？](#q-008)
[9. Jensen 不等式在机器学习推导中如何使用？](#q-009)
[10. VAE 的 ELBO 如何推导？](#q-010)

## 第三章 Transformer 数学推导

[11. Attention 为什么要除以 $\sqrt{d_k}$？](#q-011)
[12. RoPE 为什么能表达相对位置信息？](#q-012)
[13. KV Cache 为什么能降低自回归推理计算？](#q-013)
[14. FlashAttention 的 Online Softmax 如何保证精确？](#q-014)

## 第四章 扩散模型与 Flow 推导

[15. DDPM 的 $q(x_t|x_0)$ 一步采样公式如何推导？](#q-015)
[16. DDPM 的训练目标为什么可以化简为预测噪声 MSE？](#q-016)
[17. DDIM 为什么可以做确定性采样？](#q-017)
[18. Score、SDE、ODE 三者如何互相联系？](#q-018)
[19. Flow Matching 的速度场目标如何理解？](#q-019)

## 第五章 对齐学习与高效微调推导

[20. DPO Loss 如何从 KL 约束优化目标推导？](#q-020)
[21. PPO clipped objective 的数学意义是什么？](#q-021)
[22. GRPO 的组内相对优势如何计算？](#q-022)
[23. LoRA 的参数量和计算量如何推导？](#q-023)
[24. 量化 scale 和 zero-point 如何计算？](#q-024)
[25. 知识蒸馏中的温度系数为什么常配 $T^2$？](#q-025)

---

<h1 id="q-001">1. Softmax + Cross Entropy 的梯度如何推导？</h1>

设 logits 为 $z \in \mathbb{R}^{K}$，Softmax 输出为：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

交叉熵损失为：

$$
L=-\sum_{i=1}^{K}y_i\log p_i
$$

其中 $y$ 是真实标签分布。若 $y$ 是 one-hot，设真实类别为 $c$，则：

$$
L=-\log p_c
$$

先写出 Softmax 的 Jacobian：

$$
\frac{\partial p_i}{\partial z_k}=p_i(\delta_{ik}-p_k)
$$

对损失求导：

$$
\frac{\partial L}{\partial z_k}
=-\sum_i y_i\frac{1}{p_i}\frac{\partial p_i}{\partial z_k}
$$

代入 Softmax 导数：

$$
\frac{\partial L}{\partial z_k}
=-\sum_i y_i(\delta_{ik}-p_k)
$$

因为 $\sum_i y_i=1$，所以：

$$
\frac{\partial L}{\partial z_k}=p_k-y_k
$$

**结论：**

$$
\nabla_z L = p-y
$$

**面试要点：**

- Softmax 和 Cross Entropy 合并后梯度非常简洁。
- 真实类别的 logit 梯度为 $p_c-1$，非真实类别为 $p_k$。
- 工程实现通常使用 `log_softmax + nll_loss` 或 fused cross entropy，避免数值溢出。

---

<h1 id="q-002">2. Sigmoid + Binary Cross Entropy 的梯度如何推导？</h1>

二分类中，设 logit 为 $z$：

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

Binary Cross Entropy：

$$
L=-y\log p-(1-y)\log(1-p)
$$

Sigmoid 导数：

$$
\frac{dp}{dz}=p(1-p)
$$

先对 $p$ 求导：

$$
\frac{\partial L}{\partial p}
=-\frac{y}{p}+\frac{1-y}{1-p}
$$

链式法则：

$$
\frac{\partial L}{\partial z}
=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)
$$

展开：

$$
\frac{\partial L}{\partial z}
=-y(1-p)+(1-y)p
$$

整理得到：

$$
\frac{\partial L}{\partial z}=p-y
$$

**结论：**

Sigmoid + BCE 与 Softmax + CE 一样，logit 层梯度都是“预测概率 - 真实标签”。

**工程注意：**

实际训练中优先使用 `BCEWithLogitsLoss`，它把 Sigmoid 和 BCE 合并，避免 $p$ 接近 0 或 1 时出现 $\log 0$ 和数值不稳定。

---

<h1 id="q-003">3. MSE 为什么对应高斯噪声假设？</h1>

假设回归标签由真实函数加高斯噪声得到：

$$
y=f_{\theta}(x)+\epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma^2)
$$

则条件概率为：

$$
p(y|x;\theta)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-f_{\theta}(x))^2}{2\sigma^2}\right)
$$

对数据集取负对数似然：

$$
-\log p(D|\theta)
=-\sum_i \log p(y_i|x_i;\theta)
$$

代入高斯密度：

$$
-\log p(D|\theta)
=\sum_i \left[\frac{(y_i-f_{\theta}(x_i))^2}{2\sigma^2}+\frac{1}{2}\log(2\pi\sigma^2)\right]
$$

若 $\sigma$ 固定，与 $\theta$ 无关的常数项可忽略：

$$
\theta^*=\arg\min_{\theta}\sum_i (y_i-f_{\theta}(x_i))^2
$$

这就是 MSE。

**结论：**

MSE 等价于假设观测误差服从同方差高斯分布时的最大似然估计。

**面试追问：为什么 MSE 对异常值敏感？**

因为误差被平方放大，大残差会主导损失。

---

<h1 id="q-004">4. MAE 为什么对应 Laplace 噪声假设？</h1>

假设误差服从 Laplace 分布：

$$
y=f_{\theta}(x)+\epsilon,\quad \epsilon\sim\text{Laplace}(0,b)
$$

Laplace 密度：

$$
p(y|x;\theta)=\frac{1}{2b}\exp\left(-\frac{|y-f_{\theta}(x)|}{b}\right)
$$

负对数似然：

$$
-\log p(D|\theta)
=\sum_i\left[\log(2b)+\frac{|y_i-f_{\theta}(x_i)|}{b}\right]
$$

若 $b$ 固定，忽略常数和比例系数：

$$
\theta^*=\arg\min_{\theta}\sum_i |y_i-f_{\theta}(x_i)|
$$

这就是 MAE。

**结论：**

MAE 等价于假设观测误差服从 Laplace 分布时的最大似然估计。

**MSE 与 MAE 对比：**

- MSE 对大误差更敏感，优化更平滑。
- MAE 对异常值更鲁棒，但在 0 点不可导，实践中可用 Smooth L1/Huber Loss 折中。

Huber Loss：

$$
L_{\delta}(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\le\delta \\
\delta(|r|-\frac{1}{2}\delta), & |r|>\delta
\end{cases}
$$

---

<h1 id="q-005">5. LayerNorm 和 RMSNorm 的核心反向传播直觉是什么？</h1>

LayerNorm 对单个样本的 hidden 维度归一化：

$$
\mu=\frac{1}{d}\sum_i x_i
$$

$$
\sigma^2=\frac{1}{d}\sum_i(x_i-\mu)^2
$$

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

$$
y_i=\gamma_i\hat{x}_i+\beta_i
$$

反向传播中，每个 $x_i$ 的梯度不仅来自自己的输出，还会通过均值 $\mu$ 和方差 $\sigma^2$ 影响所有维度。因此 LayerNorm 的梯度是跨 hidden 维耦合的。

RMSNorm：

$$
\text{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}
$$

$$
y_i=\gamma_i\frac{x_i}{\text{RMS}(x)}
$$

RMSNorm 不减均值，只按均方根缩放，反向传播少了均值中心化相关项，计算更简单。

**面试回答重点：**

- LayerNorm 稳定每个 token 的 hidden 分布。
- RMSNorm 保留方向信息，只控制向量尺度。
- 两者都让残差网络中的激活尺度更稳定，改善深层 Transformer 训练。
- Pre-Norm Transformer 中归一化放在子层前，更利于梯度通过残差路径传播。

---

<h1 id="q-006">6. 最大似然估计为什么等价于最小化负对数似然？</h1>

给定独立同分布数据：

$$
D=\{x_1,x_2,...,x_N\}
$$

似然函数：

$$
L(\theta)=p(D|\theta)=\prod_{i=1}^{N}p(x_i|\theta)
$$

最大似然估计：

$$
\theta_{\text{MLE}}=\arg\max_{\theta}\prod_i p(x_i|\theta)
$$

乘积不方便计算，取对数：

$$
\log L(\theta)=\sum_i \log p(x_i|\theta)
$$

因为 $\log$ 是单调递增函数，最大化似然等价于最大化对数似然：

$$
\theta_{\text{MLE}}=\arg\max_{\theta}\sum_i\log p(x_i|\theta)
$$

深度学习优化器通常做最小化，因此写成最小化负对数似然：

$$
\theta_{\text{MLE}}=\arg\min_{\theta}-\sum_i\log p(x_i|\theta)
$$

**大模型对应：**

自回归语言模型：

$$
p_{\theta}(x)=\prod_t p_{\theta}(x_t|x_{<t})
$$

训练损失：

$$
L=-\sum_t \log p_{\theta}(x_t|x_{<t})
$$

这就是 token 级负对数似然。

---

<h1 id="q-007">7. Cross Entropy、KL 散度和最大似然之间有什么关系？</h1>

交叉熵：

$$
H(P,Q)=-\sum_x P(x)\log Q(x)
$$

KL 散度：

$$
D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

展开 KL：

$$
D_{KL}(P\|Q)=\sum_x P(x)\log P(x)-\sum_x P(x)\log Q(x)
$$

因为：

$$
H(P)=-\sum_xP(x)\log P(x)
$$

所以：

$$
D_{KL}(P\|Q)=H(P,Q)-H(P)
$$

即：

$$
H(P,Q)=H(P)+D_{KL}(P\|Q)
$$

当真实分布 $P$ 固定时，$H(P)$ 与模型无关，最小化交叉熵等价于最小化 $D_{KL}(P\|Q)$。

经验分布下：

$$
\hat{P}(x)=\frac{1}{N}\sum_i\delta(x-x_i)
$$

最小化交叉熵：

$$
-\frac{1}{N}\sum_i\log q_{\theta}(x_i)
$$

等价于最大似然估计。

**一句话总结：**

最大似然、交叉熵训练、最小化真实分布到模型分布的 KL，在监督学习和语言模型预训练中本质上是同一件事的不同表述。

---

<h1 id="q-008">8. KL 散度非负性如何证明？</h1>

KL 散度：

$$
D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}
$$

使用 Jensen 不等式。因为 $-\log x$ 是凸函数：

$$
D_{KL}(P\|Q)
=\sum_xP(x)\left[-\log\frac{Q(x)}{P(x)}\right]
$$

由 Jensen 不等式：

$$
\sum_xP(x)\left[-\log\frac{Q(x)}{P(x)}\right]
\ge
-\log\sum_xP(x)\frac{Q(x)}{P(x)}
$$

右边化简：

$$
-\log\sum_xQ(x)=-\log 1=0
$$

因此：

$$
D_{KL}(P\|Q)\ge 0
$$

当且仅当 $P=Q$ 时取等号。

**面试易错点：**

- KL 非负，但不是距离，因为它不对称，也不满足三角不等式。
- 若存在 $P(x)>0$ 但 $Q(x)=0$，KL 会发散到无穷大。

---

<h1 id="q-009">9. Jensen 不等式在机器学习推导中如何使用？</h1>

Jensen 不等式：

若 $f$ 是凸函数，则：

$$
f(\mathbb{E}[X])\le \mathbb{E}[f(X)]
$$

若 $f$ 是凹函数，则：

$$
f(\mathbb{E}[X])\ge \mathbb{E}[f(X)]
$$

机器学习中常见用途：

**1. 证明 KL 非负**

使用 $-\log x$ 的凸性。

**2. 推导 VAE 的 ELBO**

因为 $\log$ 是凹函数：

$$
\log \mathbb{E}[X]\ge \mathbb{E}[\log X]
$$

可把难以直接优化的 $\log p_{\theta}(x)$ 转成可优化下界。

**3. 分析期望损失**

例如凸损失下，平均预测与平均损失之间存在 Jensen 关系。

**面试回答模板：**

看到“对数里面有积分/求和/期望”，通常可以考虑用 Jensen 不等式把 log 与 expectation 交换，从而得到上界或下界。

---

<h1 id="q-010">10. VAE 的 ELBO 如何推导？</h1>

VAE 要最大化边缘似然：

$$
\log p_{\theta}(x)=\log\int p_{\theta}(x,z)dz
$$

引入近似后验 $q_{\phi}(z|x)$：

$$
\log p_{\theta}(x)
=\log\int q_{\phi}(z|x)\frac{p_{\theta}(x,z)}{q_{\phi}(z|x)}dz
$$

写成期望：

$$
\log p_{\theta}(x)
=\log \mathbb{E}_{q_{\phi}(z|x)}\left[\frac{p_{\theta}(x,z)}{q_{\phi}(z|x)}\right]
$$

由于 $\log$ 是凹函数，由 Jensen 不等式：

$$
\log p_{\theta}(x)
\ge
\mathbb{E}_{q_{\phi}(z|x)}
\left[
\log\frac{p_{\theta}(x,z)}{q_{\phi}(z|x)}
\right]
$$

右边就是 ELBO：

$$
\mathcal{L}_{\text{ELBO}}
=
\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]
-D_{KL}(q_{\phi}(z|x)\|p(z))
$$

**两项含义：**

- 重构项：让 $z$ 能生成原始数据 $x$。
- KL 正则：让近似后验接近先验，保证潜空间可采样。

**与扩散/Stable Diffusion 的联系：**

Latent Diffusion 先用 VAE 把图像压缩到潜空间，再在潜空间做扩散建模。VAE 的质量会影响潜空间信息保真上限。

---

<h1 id="q-011">11. Attention 为什么要除以 $\sqrt{d_k}$？</h1>

假设 query 和 key 的每个分量独立，均值为 0，方差为 1：

$$
q_i,k_i \sim (0,1)
$$

点积：

$$
q\cdot k=\sum_{i=1}^{d_k}q_ik_i
$$

因为：

$$
\mathbb{E}[q_ik_i]=0
$$

且在独立假设下：

$$
\text{Var}(q_ik_i)=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=1
$$

所以：

$$
\text{Var}(q\cdot k)=d_k
$$

点积标准差为：

$$
\sqrt{d_k}
$$

若不缩放，$d_k$ 越大，attention logits 绝对值越大，Softmax 越容易饱和，梯度越小。

缩放后：

$$
\frac{q\cdot k}{\sqrt{d_k}}
$$

方差约为 1，训练更稳定。

**结论：**

除以 $\sqrt{d_k}$ 不是经验魔法，而是为了控制点积方差，使 Softmax 工作在更稳定的区间。

---

<h1 id="q-012">12. RoPE 为什么能表达相对位置信息？</h1>

RoPE 在二维子空间中对位置 $m$ 的向量做旋转：

$$
R(m\theta)=
\begin{bmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{bmatrix}
$$

对 query 和 key 分别注入位置：

$$
\tilde{q}_m=R(m\theta)q
$$

$$
\tilde{k}_n=R(n\theta)k
$$

注意力点积：

$$
\tilde{q}_m^T\tilde{k}_n
=q^TR(m\theta)^TR(n\theta)k
$$

旋转矩阵满足：

$$
R(a)^T=R(-a)
$$

所以：

$$
R(m\theta)^TR(n\theta)=R((n-m)\theta)
$$

因此：

$$
\tilde{q}_m^T\tilde{k}_n
=q^TR((n-m)\theta)k
$$

点积只依赖相对位置 $n-m$，而不是绝对位置 $m,n$。

**结论：**

RoPE 通过旋转矩阵的群结构，把绝对位置注入到 $Q,K$ 中，但注意力分数天然包含相对位置信息。

---

<h1 id="q-013">13. KV Cache 为什么能降低自回归推理计算？</h1>

自回归生成第 $t$ 个 token 时：

$$
p(x_t|x_{<t})
$$

每层注意力需要：

$$
Q_t = h_tW_Q
$$

$$
K_{1:t}=h_{1:t}W_K
$$

$$
V_{1:t}=h_{1:t}W_V
$$

如果没有 KV Cache，每生成一个新 token，都要重新计算所有历史 token 的 $K,V$。

但历史 token 的 hidden state 在自回归推理中已经确定，对当前步不会变化，因此历史 $K,V$ 可缓存：

$$
K_{\text{cache}}=[K_1,...,K_{t-1}]
$$

$$
V_{\text{cache}}=[V_1,...,V_{t-1}]
$$

当前步只计算：

$$
K_t,V_t,Q_t
$$

并与缓存拼接。

**复杂度直觉：**

- 无缓存：每步重复处理长度 $t$ 的前缀，总计算近似 $O(T^2)$ 次投影。
- 有缓存：每步只对新增 token 做投影，投影部分近似 $O(T)$。
- 注意力打分仍需当前 token 和所有历史 key 交互，所以每步 attention 仍随上下文长度增长。

**工程代价：**

KV Cache 显著加速推理，但显存占用随层数、batch、上下文长度、KV head 数线性增长。

---

<h1 id="q-014">14. FlashAttention 的 Online Softmax 如何保证精确？</h1>

标准 Softmax 为：

$$
\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

为数值稳定，通常减去最大值：

$$
\text{softmax}(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\quad m=\max_jx_j
$$

FlashAttention 分块处理 attention logits，不能一次性拿到完整行的所有 $x_j$，因此使用 online softmax 维护当前块之前的最大值和归一化分母。

假设已处理部分的最大值为 $m_{\text{old}}$，分母为 $l_{\text{old}}$：

$$
l_{\text{old}}=\sum_{\text{old}}e^{x_j-m_{\text{old}}}
$$

新块最大值为 $m_{\text{block}}$，新全局最大值：

$$
m_{\text{new}}=\max(m_{\text{old}},m_{\text{block}})
$$

旧分母需要换基准：

$$
l_{\text{old}}' = l_{\text{old}}e^{m_{\text{old}}-m_{\text{new}}}
$$

新块分母：

$$
l_{\text{block}}'=\sum_{\text{block}}e^{x_j-m_{\text{new}}}
$$

合并：

$$
l_{\text{new}}=l_{\text{old}}'+l_{\text{block}}'
$$

输出加权和也按同样比例重标定。因此分块计算能得到与完整 Softmax 等价的结果。

**结论：**

FlashAttention 不是近似注意力。它通过分块、重计算和 online softmax 降低显存读写，数学结果仍是精确 attention。

---

<h1 id="q-015">15. DDPM 的 $q(x_t|x_0)$ 一步采样公式如何推导？</h1>

DDPM 前向过程：

$$
q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

其中：

$$
\alpha_t=1-\beta_t
$$

可重参数化为：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

对 $x_{t-1}$ 继续展开：

$$
x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-1}
$$

代入得到：

$$
x_t=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\text{Gaussian noise}
$$

由于独立高斯噪声的线性组合仍是高斯噪声，递推可得：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

其中：

$$
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s,\quad \epsilon\sim\mathcal{N}(0,I)
$$

因此：

$$
q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

**意义：**

训练时不需要一步步加噪到 $t$，可以随机采样 $t$ 后直接生成 $x_t$，极大提高训练效率。

---

<h1 id="q-016">16. DDPM 的训练目标为什么可以化简为预测噪声 MSE？</h1>

DDPM 反向过程学习：

$$
p_{\theta}(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_{\theta}(x_t,t),\sigma_t^2I)
$$

变分下界中包含真实后验和模型反向分布之间的 KL：

$$
D_{KL}(q(x_{t-1}|x_t,x_0)\|p_{\theta}(x_{t-1}|x_t))
$$

两个高斯分布方差固定时，KL 关于均值的部分等价于加权 MSE：

$$
\|\tilde{\mu}_t(x_t,x_0)-\mu_{\theta}(x_t,t)\|_2^2
$$

真实后验均值可写为 $x_t$ 和 $x_0$ 的函数，而 $x_0$ 又可由噪声表达：

$$
x_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
$$

因此模型预测均值可以重参数化为预测噪声：

$$
\mu_{\theta}(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{\theta}(x_t,t)
\right)
$$

忽略与时间步相关的权重后，得到简化目标：

$$
L_{\text{simple}}=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\|\epsilon-\epsilon_{\theta}(x_t,t)\|_2^2
\right]
$$

**结论：**

DDPM 不是随意选择预测噪声，而是因为高斯后验 KL 在特定参数化下可以化简为噪声预测 MSE，训练稳定且效果好。

---

<h1 id="q-017">17. DDIM 为什么可以做确定性采样？</h1>

DDPM 采样通常包含随机噪声项：

$$
x_{t-1}=\mu_{\theta}(x_t,t)+\sigma_t z,\quad z\sim\mathcal{N}(0,I)
$$

DDIM 构造了一个非马尔可夫反向过程，使得边缘分布 $q(x_t|x_0)$ 与 DDPM 保持一致，但采样路径可以由参数 $\eta$ 控制随机性。

常见形式：

$$
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}\hat{x}_0
+
\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_{\theta}(x_t,t)
+
\sigma_t z
$$

其中：

$$
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_{\theta}(x_t,t)}{\sqrt{\bar{\alpha}_t}}
$$

当 $\eta=0$ 时：

$$
\sigma_t=0
$$

于是：

$$
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}\hat{x}_0
+
\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_{\theta}(x_t,t)
$$

不再注入随机噪声，因此给定初始噪声 $x_T$ 和条件，采样轨迹是确定的。

**意义：**

- DDIM 可以少步采样。
- 确定性路径便于图像反演和编辑。
- 随机性由 $\eta$ 控制，$\eta=0$ 是确定性，$\eta>0$ 更接近随机采样。

---

<h1 id="q-018">18. Score、SDE、ODE 三者如何互相联系？</h1>

Score 定义为：

$$
s_t(x)=\nabla_x\log p_t(x)
$$

它表示当前噪声尺度下，对数密度对样本的梯度。

连续扩散正向过程可写为 SDE：

$$
dx=f(x,t)dt+g(t)dw
$$

反向时间 SDE 为：

$$
dx=[f(x,t)-g(t)^2\nabla_x\log p_t(x)]dt+g(t)d\bar{w}
$$

其中 $\nabla_x\log p_t(x)$ 就是 score，通常由神经网络学习。

对应的 Probability Flow ODE：

$$
dx=\left[f(x,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\right]dt
$$

它与反向 SDE 具有相同的边缘分布演化，但采样路径是确定性的。

**关系总结：**

- Score：告诉模型如何朝高密度区域移动。
- SDE：随机微分方程，包含噪声项。
- ODE：确定性流，去掉随机项但保持相同边缘分布。
- DDPM 是离散时间版本，Score SDE 是连续统一框架。

---

<h1 id="q-019">19. Flow Matching 的速度场目标如何理解？</h1>

Flow Matching 希望学习一个速度场：

$$
v_{\theta}(x,t)
$$

使样本从简单分布 $p_0$ 流动到数据分布 $p_1$。

设 $x_0\sim p_0$ 是噪声，$x_1\sim p_1$ 是数据。若使用线性路径：

$$
x_t=(1-t)x_0+tx_1
$$

则路径对时间求导：

$$
\frac{dx_t}{dt}=x_1-x_0
$$

因此目标速度可以写为：

$$
u_t=x_1-x_0
$$

训练目标：

$$
L_{\text{FM}}=
\mathbb{E}_{t,x_t}
\left[
\|v_{\theta}(x_t,t)-u_t(x_t)\|_2^2
\right]
$$

采样时从噪声出发，求解 ODE：

$$
\frac{dx}{dt}=v_{\theta}(x,t)
$$

从 $t=0$ 积分到 $t=1$ 得到数据样本。

**与扩散模型对比：**

- 扩散模型常学习 score 或噪声。
- Flow Matching 学习速度场。
- Rectified Flow 希望路径更直，让少步采样更高效。

---

<h1 id="q-020">20. DPO Loss 如何从 KL 约束优化目标推导？</h1>

RLHF 的理想目标可以写成在 KL 约束下最大化奖励：

$$
\max_{\pi}
\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]
-
\beta D_{KL}(\pi(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))
$$

该目标的最优策略满足：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

两边取对数并整理：

$$
r(x,y)=
\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}
+
\beta\log Z(x)
$$

对于同一个 prompt 下的两个回答 $y_w,y_l$，归一化项 $\log Z(x)$ 抵消：

$$
r(x,y_w)-r(x,y_l)
=
\beta
\left[
\log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
$$

偏好模型使用 Bradley-Terry 形式：

$$
P(y_w \succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))
$$

用当前策略 $\pi_{\theta}$ 近似 $\pi^*$，得到 DPO 损失：

$$
L_{\text{DPO}}
=
-\log\sigma
\left(
\beta
\left[
\log\frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log\frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
$$

**一句话：**

DPO 把 KL 约束 RL 问题转成了偏好对上的二分类问题，避免显式训练 reward model 和在线 PPO。

---

<h1 id="q-021">21. PPO clipped objective 的数学意义是什么？</h1>

PPO 定义新旧策略概率比：

$$
r_t(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

普通策略梯度目标可写为：

$$
L(\theta)=\mathbb{E}[r_t(\theta)A_t]
$$

如果 $r_t$ 变化过大，策略更新会过猛，导致训练不稳定。PPO 使用裁剪目标：

$$
L^{\text{CLIP}}(\theta)=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

理解方式：

- 当 $A_t>0$，说明动作好，希望提高概率，但最多提高到 $1+\epsilon$ 附近。
- 当 $A_t<0$，说明动作差，希望降低概率，但最多降低到 $1-\epsilon$ 附近。

**数学意义：**

PPO clipped objective 是一种简单的 trust region 近似，用概率比限制策略每步变化幅度，降低策略崩坏风险。

---

<h1 id="q-022">22. GRPO 的组内相对优势如何计算？</h1>

GRPO 对同一个 prompt 采样一组回答：

$$
\{y_1,y_2,...,y_G\}
$$

对每个回答计算奖励：

$$
\{r_1,r_2,...,r_G\}
$$

组内均值：

$$
\mu_r=\frac{1}{G}\sum_{i=1}^{G}r_i
$$

组内标准差：

$$
\sigma_r=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu_r)^2+\epsilon}
$$

相对优势：

$$
A_i=\frac{r_i-\mu_r}{\sigma_r}
$$

之后将 $A_i$ 放入类似 PPO 的 clipped policy objective，并加入 KL 约束。

**为什么这样做？**

- 不需要单独训练 value model。
- 同题多答案之间的相对好坏更容易估计。
- 对数学、代码、推理任务，奖励常来自规则判分或自动验证，适合组内比较。

**注意：**

如果一组回答奖励几乎相同，$\sigma_r$ 很小，需要加 $\epsilon$ 防止数值不稳定。

---

<h1 id="q-023">23. LoRA 的参数量和计算量如何推导？</h1>

原始线性层：

$$
y=Wx
$$

其中：

$$
W\in\mathbb{R}^{d_{out}\times d_{in}}
$$

原始参数量：

$$
d_{out}d_{in}
$$

LoRA 冻结 $W$，学习：

$$
\Delta W=BA
$$

其中：

$$
B\in\mathbb{R}^{d_{out}\times r},\quad A\in\mathbb{R}^{r\times d_{in}}
$$

LoRA 可训练参数量：

$$
rd_{out}+rd_{in}=r(d_{out}+d_{in})
$$

当 $r\ll \min(d_{out},d_{in})$ 时，参数量大幅减少。

前向计算：

$$
y=Wx+\frac{\alpha}{r}BAx
$$

LoRA 分支计算量：

1. 先算 $Ax$：约 $rd_{in}$。
2. 再算 $B(Ax)$：约 $d_{out}r$。

总额外计算量：

$$
O(r(d_{in}+d_{out}))
$$

而完整更新矩阵的计算量为：

$$
O(d_{in}d_{out})
$$

**面试结论：**

LoRA 的高效来自低秩假设：下游任务所需权重变化 $\Delta W$ 位于低维子空间。

---

<h1 id="q-024">24. 量化 scale 和 zero-point 如何计算？</h1>

量化把浮点数 $x$ 映射到整数 $q$。

非对称均匀量化设浮点范围为：

$$
[x_{\min},x_{\max}]
$$

整数范围为：

$$
[q_{\min},q_{\max}]
$$

scale：

$$
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}}
$$

zero-point：

$$
z=q_{\min}-\text{round}\left(\frac{x_{\min}}{s}\right)
$$

量化：

$$
q=\text{clip}\left(\text{round}\left(\frac{x}{s}\right)+z,q_{\min},q_{\max}\right)
$$

反量化：

$$
\hat{x}=s(q-z)
$$

对称量化通常令 zero-point 为 0：

$$
q=\text{round}\left(\frac{x}{s}\right)
$$

$$
\hat{x}=sq
$$

**误差来源：**

$$
e=x-\hat{x}
$$

均匀量化中，未裁剪情况下误差通常被限制在半个量化间隔附近：

$$
|e|\le \frac{s}{2}
$$

**LLM 量化要点：**

- per-channel 或 group-wise scale 通常比 per-tensor 更准。
- outlier channel 会显著影响 scale，导致大多数普通值量化精度下降。
- NF4 使用非均匀码本，更适合近似正态分布的权重。

---

<h1 id="q-025">25. 知识蒸馏中的温度系数为什么常配 $T^2$？</h1>

蒸馏使用带温度 Softmax：

$$
p_i^{(T)}=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

蒸馏损失：

$$
L_{\text{KD}}=D_{KL}(p_T^{\text{teacher}}\|p_T^{\text{student}})
$$

温度 $T$ 变大时，Softmax 分布变平滑，logits 对概率的影响被缩小。对 student logits 求导时，Softmax 中的 $z/T$ 会带来一个 $\frac{1}{T}$ 因子。

同时，当 $T$ 较大时，teacher 与 student 的概率差也近似随 $\frac{1}{T}$ 缩小。因此梯度量级大约会随：

$$
\frac{1}{T^2}
$$

变小。

为了让不同温度下蒸馏损失的梯度量级可比，常把 KL 损失乘以：

$$
T^2
$$

即：

$$
L_{\text{KD}}=T^2D_{KL}(p_T^{\text{teacher}}\|p_T^{\text{student}})
$$

**面试回答：**

$T$ 用来软化分布，暴露类别之间的暗知识；乘 $T^2$ 主要是补偿温度导致的梯度缩放，使蒸馏项和监督项在训练中保持合理权重。

---

## 参考资料

[1] Ian Goodfellow, Yoshua Bengio, Aaron Courville. *Deep Learning*. MIT Press, 2016.

[2] Vaswani et al. *Attention Is All You Need*. NeurIPS, 2017. <https://arxiv.org/abs/1706.03762>

[3] Ho et al. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020. <https://arxiv.org/abs/2006.11239>

[4] Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR, 2021. <https://arxiv.org/abs/2011.13456>

[5] Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS, 2023. <https://arxiv.org/abs/2305.18290>

[6] Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR, 2022. <https://arxiv.org/abs/2106.09685>

[7] Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS, 2022. <https://arxiv.org/abs/2205.14135>
