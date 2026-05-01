# 目录

## 第一章 线性代数与张量基础高频考点

[1. 深度学习与大模型为什么离不开线性代数？](#q-001)
  - [面试问题：标量、向量、矩阵、张量之间有什么联系和区别？](#q-002)
  - [面试问题：矩阵乘法、批量矩阵乘法和张量维度在 Transformer 中如何理解？](#q-003)
  - [面试问题：向量范数、矩阵范数、核范数、Frobenius 范数分别有什么作用？](#q-004)
  - [面试问题：什么是正定矩阵？为什么协方差矩阵和 Hessian 经常要求半正定？](#q-005)

[2. 特征值分解、SVD 与低秩近似为什么重要？](#q-006)
  - [面试问题：特征值、特征向量的几何意义是什么？](#q-007)
  - [面试问题：奇异值分解 SVD 与特征值分解有什么关系？](#q-008)
  - [面试问题：低秩近似如何解释 PCA、LoRA、模型压缩和推荐系统？](#q-009)
  - [面试问题：线性空间、基、维度和线性变换在深度学习中如何理解？](#q-009a)

## 第二章 微积分、矩阵求导与反向传播高频考点

[3. 导数、偏导、梯度、Jacobian、Hessian 有什么区别？](#q-010)
  - [面试问题：导数和偏导数有什么区别？](#q-011)
  - [面试问题：梯度为什么指向函数上升最快方向？](#q-012)
  - [面试问题：Jacobian 和 Hessian 在深度学习中分别出现在哪里？](#q-013)
  - [面试问题：链式法则如何支撑反向传播和自动微分？](#q-014)
  - [面试问题：泰勒展开和极限思想在优化与大模型训练中有什么作用？](#q-014a)

[4. 常见激活函数、归一化和损失函数如何求导？](#q-015)
  - [面试问题：Sigmoid、Tanh、ReLU、GELU、SiLU 的导数和梯度问题是什么？](#q-016)
  - [面试问题：Softmax + Cross Entropy 为什么梯度形式很简洁？](#q-017)
  - [面试问题：LayerNorm、RMSNorm 的数学作用是什么？](#q-018)

## 第三章 概率统计与常见分布高频考点

[5. 机器学习为什么需要概率论？](#q-019)
  - [面试问题：随机变量、概率分布、PMF、PDF 分别是什么？](#q-020)
  - [面试问题：联合概率、边缘概率、条件概率、贝叶斯公式有什么联系？](#q-021)
  - [面试问题：独立性和条件独立性有什么区别？](#q-022)
  - [面试问题：期望、方差、协方差、相关系数如何理解？](#q-023)

[6. 常见概率分布在 AI 中如何使用？](#q-024)
  - [面试问题：Bernoulli、Categorical、Binomial、Multinomial 分布分别适合什么场景？](#q-025)
  - [面试问题：高斯分布、多元高斯、协方差矩阵为什么重要？](#q-026)
  - [面试问题：指数分布、Laplace 分布、Dirac 分布和经验分布有什么用途？](#q-027)
  - [面试问题：最大似然估计、MAP、贝叶斯估计有什么区别？](#q-028)

## 第四章 信息论、损失函数与评价指标高频考点

[7. 熵、交叉熵、KL 散度、互信息为什么是大模型基础？](#q-029)
  - [面试问题：熵、交叉熵和困惑度 Perplexity 如何理解？](#q-030)
  - [面试问题：KL 散度为什么不对称？在 VAE、蒸馏、RLHF 中怎么用？](#q-031)
  - [面试问题：JS 散度、互信息、最大熵原理在生成模型中有什么意义？](#q-032)

## 第五章 优化理论与训练稳定性高频考点

[8. 梯度下降与现代优化器的数学本质是什么？](#q-033)
  - [面试问题：SGD、Momentum、AdaGrad、RMSProp、Adam、AdamW 有什么区别？](#q-034)
  - [面试问题：学习率、Batch Size、梯度噪声、Warmup、Cosine Decay 如何影响训练？](#q-035)
  - [面试问题：梯度爆炸、梯度消失、梯度裁剪如何从数学上理解？](#q-036)
  - [面试问题：过拟合、正则化、Dropout、Label Smoothing 的数学作用是什么？](#q-037)
  - [面试问题：无约束优化、有约束优化、拉格朗日乘子和 KKT 条件如何理解？](#q-037a)
  - [面试问题：凸优化和非凸优化有什么区别？深度学习为什么仍能训练成功？](#q-037b)

## 第六章 Transformer 与大语言模型数学高频考点

[9. Attention 的核心数学是什么？](#q-038)
  - [面试问题：Scaled Dot-Product Attention 为什么要除以 $\sqrt{d_k}$？](#q-039)
  - [面试问题：Multi-Head Attention 的线性代数本质是什么？](#q-040)
  - [面试问题：位置编码、RoPE、ALiBi 的数学差异是什么？](#q-041)
  - [面试问题：KV Cache、FlashAttention、长上下文优化背后的数学直觉是什么？](#q-042)

[10. 大模型预训练、微调和推理中的常考数学是什么？](#q-043)
  - [面试问题：自回归语言模型的最大似然训练目标是什么？](#q-044)
  - [面试问题：Temperature、Top-k、Top-p、重复惩罚如何改变采样分布？](#q-045)
  - [面试问题：Embedding 相似度、余弦相似度、向量检索和 RAG 的数学基础是什么？](#q-046)
  - [面试问题：MoE 的门控函数、负载均衡损失和稀疏激活如何理解？](#q-047)

## 第七章 扩散模型、Flow Matching 与多模态生成数学高频考点

[11. 扩散模型的概率建模基础是什么？](#q-048)
  - [面试问题：DDPM 的前向扩散和反向去噪如何用马尔可夫链表示？](#q-049)
  - [面试问题：Score Matching、SDE、ODE 与扩散模型有什么关系？](#q-050)
  - [面试问题：Classifier-Free Guidance 的数学形式是什么？为什么 CFG scale 过大会失真？](#q-051)
  - [面试问题：Flow Matching、Rectified Flow 与传统扩散模型的训练目标有什么差异？](#q-052)

## 第八章 对齐学习、RLHF、DPO 与 GRPO 数学高频考点

[12. 大模型对齐为什么需要偏好优化和强化学习？](#q-053)
  - [面试问题：RLHF 中 reward model、PPO、KL penalty 分别解决什么问题？](#q-054)
  - [面试问题：DPO 如何把偏好学习转化为分类式损失？](#q-055)
  - [面试问题：GRPO 与 PPO 的数学差异是什么？为什么适合推理模型训练？](#q-056)

## 第九章 参数高效微调、量化与部署数学高频考点

[13. LoRA、QLoRA、量化和蒸馏背后的数学是什么？](#q-057)
  - [面试问题：LoRA 为什么可以用低秩矩阵近似权重更新？](#q-058)
  - [面试问题：量化、反量化、对称/非对称量化、NF4 的核心数学是什么？](#q-059)
  - [面试问题：知识蒸馏、温度系数和 KL 损失如何理解？](#q-060)

## 第十章 面试速查与学习建议

[14. AIGC 数学基础应该如何复习？](#q-061)
  - [面试问题：算法岗、大模型工程岗、多模态方向各自最该优先掌握哪些数学？](#q-062)

---

<h1 id="q-001">1. 深度学习与大模型为什么离不开线性代数？</h1>

线性代数是深度学习的“数据表示语言”。图像、文本、音频、视频、用户行为、模型参数、梯度和优化状态最终都会被表示为向量、矩阵或高阶张量。神经网络中的全连接层、卷积、注意力、归一化、Embedding 检索、LoRA 低秩微调和量化部署，本质上都在做大规模线性变换、矩阵乘法、范数约束和低秩结构利用。

<h2 id="q-002">面试问题：标量、向量、矩阵、张量之间有什么联系和区别？</h2>

**标量（Scalar）** 是一个单独的数，可以看作 0 阶张量，例如损失值 $L$、学习率 $\eta$、温度系数 $T$。

**向量（Vector）** 是一组有序数，可以看作 1 阶张量，例如一个 token 的 embedding 向量 $x \in \mathbb{R}^{d}$。

**矩阵（Matrix）** 是二维数组，可以看作 2 阶张量，例如线性层权重 $W \in \mathbb{R}^{d_{in} \times d_{out}}$、注意力矩阵 $A \in \mathbb{R}^{n \times n}$。

**张量（Tensor）** 是更一般的多维数组。例如 LLM 中一批 token hidden states 常写为：

$$
X \in \mathbb{R}^{B \times S \times D}
$$

其中 $B$ 是 batch size，$S$ 是序列长度，$D$ 是 hidden size。多头注意力中常变形为：

$$
Q,K,V \in \mathbb{R}^{B \times H \times S \times d_h}
$$

其中 $H$ 是注意力头数，$d_h = D/H$。

**面试回答要点：**

- 标量、向量、矩阵都是张量的特殊情况。
- 深度学习中“维度”既可能指数学维度，也可能指张量 shape，需要结合上下文说明。
- 工程中最容易出错的是 batch 维、sequence 维、head 维、feature 维的排列。

<h2 id="q-003">面试问题：矩阵乘法、批量矩阵乘法和张量维度在 Transformer 中如何理解？</h2>

矩阵乘法可以理解为“线性组合”。若：

$$
A \in \mathbb{R}^{m \times n}, \quad B \in \mathbb{R}^{n \times p}
$$

则：

$$
C = AB \in \mathbb{R}^{m \times p}, \quad C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
$$

Transformer 中常见的线性层是：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

若 $X \in \mathbb{R}^{B \times S \times D}$，$W_Q \in \mathbb{R}^{D \times D}$，则 $Q \in \mathbb{R}^{B \times S \times D}$。多头拆分后：

$$
Q \in \mathbb{R}^{B \times H \times S \times d_h}
$$

注意力分数是批量矩阵乘法：

$$
S = \frac{QK^T}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times S \times S}
$$

再经过 Softmax 得到注意力权重，乘以 $V$ 得到输出：

$$
O = \text{softmax}(S)V
$$

**常见追问：为什么注意力复杂度是 $O(S^2D)$？**

因为每个 head 都要计算长度为 $S$ 的 token 两两相似度，注意力矩阵大小是 $S \times S$。当上下文长度变长时，注意力矩阵显存和计算都会成为瓶颈。

<h2 id="q-004">面试问题：向量范数、矩阵范数、核范数、Frobenius 范数分别有什么作用？</h2>

范数用于度量“大小”。常见向量范数：

$$
\|x\|_1 = \sum_i |x_i|
$$

$$
\|x\|_2 = \sqrt{\sum_i x_i^2}
$$

$$
\|x\|_{\infty} = \max_i |x_i|
$$

$$
\|x\|_p = \left(\sum_i |x_i|^p \right)^{1/p}
$$

常见矩阵范数：

$$
\|A\|_1 = \max_j \sum_i |a_{ij}|
$$

$$
\|A\|_{\infty} = \max_i \sum_j |a_{ij}|
$$

$$
\|A\|_F = \sqrt{\sum_i \sum_j a_{ij}^2}
$$

$$
\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^TA)}
$$

核范数是奇异值之和：

$$
\|A\|_* = \sum_i \sigma_i
$$

**AI 中的典型用途：**

- $L_1$ 范数：鼓励稀疏，可用于特征选择、稀疏正则。
- $L_2$ 范数：权重衰减、距离度量、Embedding 归一化。
- Frobenius 范数：矩阵参数的整体大小，常用于权重正则、低秩近似误差。
- 谱范数：约束 Lipschitz 常数，提升生成模型或判别器稳定性。
- 核范数：秩函数的凸松弛，用于低秩约束和矩阵补全。
- 梯度范数：判断梯度爆炸、裁剪梯度、监控训练稳定性。

<h2 id="q-005">面试问题：什么是正定矩阵？为什么协方差矩阵和 Hessian 经常要求半正定？</h2>

对称矩阵 $A \in \mathbb{R}^{n \times n}$ 若对任意非零向量 $x$ 都有：

$$
x^TAx > 0
$$

则称 $A$ 为正定矩阵；若：

$$
x^TAx \ge 0
$$

则称 $A$ 为半正定矩阵。

常见判定方式：

- 所有特征值均大于 0，则正定。
- 所有特征值均大于等于 0，则半正定。
- 正定矩阵的所有顺序主子式大于 0。
- 若存在满秩矩阵 $C$ 使 $A=C^TC$，则 $A$ 半正定；若 $C$ 可逆，则正定。

**为什么协方差矩阵半正定？**

协方差矩阵 $\Sigma = \mathbb{E}[(X-\mu)(X-\mu)^T]$，对任意 $v$：

$$
v^T\Sigma v = \mathbb{E}[v^T(X-\mu)(X-\mu)^Tv] = \mathbb{E}[(v^T(X-\mu))^2] \ge 0
$$

所以协方差矩阵一定半正定。

**为什么 Hessian 和优化有关？**

Hessian 矩阵 $H$ 描述函数局部曲率。若在某点梯度为 0 且 Hessian 正定，该点通常是严格局部极小值；若 Hessian 有负特征值，则存在下降方向。

---

<h1 id="q-006">2. 特征值分解、SVD 与低秩近似为什么重要？</h1>

特征值分解和 SVD 是理解“矩阵如何变换空间”的核心工具。PCA、推荐系统矩阵分解、LoRA 低秩更新、权重量化误差分析、Embedding 降维、注意力矩阵谱分析都离不开它们。

<h2 id="q-007">面试问题：特征值、特征向量的几何意义是什么？</h2>

若方阵 $A$ 存在非零向量 $v$ 和标量 $\lambda$，满足：

$$
Av = \lambda v
$$

则 $v$ 是 $A$ 的特征向量，$\lambda$ 是对应特征值。

几何意义：矩阵 $A$ 对向量 $v$ 做线性变换后，$v$ 的方向不变，只被缩放了 $\lambda$ 倍。特征值大小表示该方向被拉伸或压缩的程度。

若 $A$ 可对角化：

$$
A = Q\Lambda Q^{-1}
$$

其中 $Q$ 的列是特征向量，$\Lambda$ 是特征值构成的对角矩阵。

**机器学习直觉：**

- 大特征值方向通常解释数据变化最大的方向。
- PCA 就是寻找协方差矩阵最大特征值对应的方向。
- Hessian 的特征值刻画 loss landscape 的曲率，特征值越大，对该方向参数扰动越敏感。

<h2 id="q-008">面试问题：奇异值分解 SVD 与特征值分解有什么关系？</h2>

任意矩阵 $A \in \mathbb{R}^{m \times n}$ 都可以做奇异值分解：

$$
A = U\Sigma V^T
$$

其中：

- $U$：左奇异向量。
- $V$：右奇异向量。
- $\Sigma$：非负奇异值构成的对角矩阵。

SVD 与特征值分解的关系：

$$
A^TA V = V\Sigma^2
$$

$$
AA^T U = U\Sigma^2
$$

因此，奇异值满足：

$$
\sigma_i = \sqrt{\lambda_i(A^TA)}
$$

**为什么 SVD 比特征值分解更通用？**

特征值分解要求矩阵是方阵，并且最好可对角化；SVD 可用于任意实矩阵，是最稳定、最常用的矩阵分解之一。

<h2 id="q-009">面试问题：低秩近似如何解释 PCA、LoRA、模型压缩和推荐系统？</h2>

若只保留最大的 $r$ 个奇异值：

$$
A \approx U_r\Sigma_rV_r^T
$$

这是 Frobenius 范数和谱范数意义下最优的 rank-$r$ 近似。

**PCA：** 对中心化数据矩阵做 SVD，保留最大奇异值对应方向，相当于保留方差最大的低维子空间。

**推荐系统：** 用户-物品矩阵通常稀疏且低秩，可分解为用户隐向量和物品隐向量。

**LoRA：** 冻结原权重 $W$，只学习低秩更新：

$$
W' = W + \Delta W,\quad \Delta W = BA
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，且 $r \ll \min(d,k)$。这假设微调所需的权重变化主要落在低维子空间中。

**模型压缩：** 对权重矩阵做低秩分解，可减少参数量和矩阵乘法开销，但可能损失表达能力。

<h2 id="q-009a">面试问题：线性空间、基、维度和线性变换在深度学习中如何理解？</h2>

线性空间是满足向量加法和数乘封闭性的集合。若一个集合中的元素可以相加、可以乘以标量，并且结果仍在集合中，就可以把它看作线性空间。

一组向量 $\{v_1,...,v_k\}$ 的线性组合为：

$$
x=\sum_{i=1}^{k}a_iv_i
$$

若空间中任意向量都能由这组向量线性表示，则称它们张成该空间；若这些向量还线性无关，则它们构成一组基。基向量个数就是空间维度。

线性变换 $T$ 满足：

$$
T(ax+by)=aT(x)+bT(y)
$$

在有限维空间中，线性变换可以由矩阵表示：

$$
y=Wx
$$

**深度学习直觉：**

- Embedding 空间可以看作语义向量空间，token、图像 patch、文本句子都被映射到其中。
- 线性层是对表示空间做基变换、旋转、缩放、投影或子空间组合。
- Attention 中的 $W_Q,W_K,W_V$ 是把同一 hidden state 投影到不同子空间。
- PCA、LoRA、低秩压缩都在利用“有用变化集中在低维子空间”这一假设。
- 表征学习的核心之一，就是学习一个让任务更容易线性分离或更容易检索的空间。

**面试易错点：**

线性层本身只是线性变换，深度网络的强表达能力来自多层线性变换与非线性激活、归一化、注意力路由和残差结构的组合。

---

<h1 id="q-010">3. 导数、偏导、梯度、Jacobian、Hessian 有什么区别？</h1>

微积分描述函数变化，优化算法利用这些变化信息调整模型参数。深度学习训练本质上是：定义损失函数，用链式法则计算梯度，再用优化器更新参数。

<h2 id="q-011">面试问题：导数和偏导数有什么区别？</h2>

一元函数 $y=f(x)$ 的导数定义为：

$$
f'(x_0)=\lim_{\Delta x \to 0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}
$$

它表示函数在该点的瞬时变化率，几何意义是曲线切线斜率。

多元函数 $z=f(x,y)$ 的偏导数表示只让一个变量变化、其余变量固定时的变化率：

$$
\frac{\partial f}{\partial x}(x_0,y_0)=\lim_{\Delta x \to 0}\frac{f(x_0+\Delta x,y_0)-f(x_0,y_0)}{\Delta x}
$$

**区别总结：**

- 导数用于一元函数。
- 偏导用于多元函数。
- 偏导计算时，把其他变量看作常数。
- 神经网络参数很多，因此训练时求的是对每个参数的偏导数组成的梯度。

<h2 id="q-012">面试问题：梯度为什么指向函数上升最快方向？</h2>

多元函数 $f(x)$ 的梯度是各个偏导数组成的向量：

$$
\nabla f(x)=\left[\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2},...,\frac{\partial f}{\partial x_n}\right]^T
$$

沿单位方向 $u$ 的方向导数为：

$$
D_uf(x)=\nabla f(x)^Tu
$$

由 Cauchy-Schwarz 不等式：

$$
\nabla f(x)^Tu \le \|\nabla f(x)\|_2\|u\|_2 = \|\nabla f(x)\|_2
$$

当 $u$ 与 $\nabla f(x)$ 同方向时取最大值。因此梯度方向是函数局部上升最快方向，负梯度方向是局部下降最快方向。

<h2 id="q-013">面试问题：Jacobian 和 Hessian 在深度学习中分别出现在哪里？</h2>

若 $f:\mathbb{R}^n \to \mathbb{R}^m$，Jacobian 矩阵为：

$$
J_{ij}=\frac{\partial f_i}{\partial x_j}
$$

它描述向量函数输出对输入的局部线性变化。反向传播中，每一层都在乘局部 Jacobian 或其转置。

若 $f:\mathbb{R}^n \to \mathbb{R}$，Hessian 矩阵为：

$$
H_{ij}=\frac{\partial^2 f}{\partial x_i \partial x_j}
$$

它描述二阶曲率。二阶优化、Newton 法、损失曲面分析、Sharpness、Fisher 信息近似都与 Hessian 相关。

**工程面试常考点：**

- 完整 Hessian 太大，实际很少显式构造。
- 常用 Hessian-vector product、Fisher 近似、K-FAC、低秩近似。
- Adam 并不是二阶优化器，而是用一阶梯度的一阶矩和二阶矩做自适应缩放。

<h2 id="q-014">面试问题：链式法则如何支撑反向传播和自动微分？</h2>

若：

$$
y=f(g(x))
$$

则：

$$
\frac{dy}{dx}=\frac{df}{dg}\frac{dg}{dx}
$$

神经网络可以看作复合函数：

$$
L = f_n(f_{n-1}(...f_1(x;\theta_1)...;\theta_{n-1});\theta_n)
$$

反向传播就是从损失 $L$ 出发，沿计算图反向应用链式法则，把梯度传给每个参数。

**自动微分的核心：**

- 前向模式适合输入维度小、输出维度大的场景。
- 反向模式适合输出是标量 loss、参数维度很大的深度学习场景。
- PyTorch、TensorFlow 等框架记录计算图，并按拓扑顺序反向传播梯度。

<h2 id="q-014a">面试问题：泰勒展开和极限思想在优化与大模型训练中有什么作用？</h2>

泰勒展开用函数在某点的导数信息近似局部函数形状。一阶泰勒展开：

$$
f(x+\Delta x)\approx f(x)+\nabla f(x)^T\Delta x
$$

二阶泰勒展开：

$$
f(x+\Delta x)\approx f(x)+\nabla f(x)^T\Delta x+\frac{1}{2}\Delta x^TH\Delta x
$$

其中 $H$ 是 Hessian。

**在优化中的意义：**

- 梯度下降来自一阶近似：沿负梯度方向能最快降低局部线性近似。
- Newton 法来自二阶近似：利用 Hessian 曲率信息选择更新方向。
- 学习率过大时，局部近似失效，训练可能震荡或发散。
- Sharpness、平坦极小值、Hessian 特征值分析都依赖二阶局部近似。

极限思想贯穿深度学习数学：

- 导数是变化率在步长趋于 0 时的极限。
- 梯度下降可看作连续梯度流的离散近似。
- SDE/ODE 扩散模型把离散去噪过程推广到连续时间极限。
- 大 batch 梯度趋近全量梯度，小学习率更新趋近连续优化轨迹。

**面试回答模板：**

泰勒展开告诉我们“局部怎么变”，梯度给出一阶方向，Hessian 给出二阶曲率；现代深度学习主要用一阶方法，是因为参数规模太大，完整二阶矩阵难以存储和求逆。

---

<h1 id="q-015">4. 常见激活函数、归一化和损失函数如何求导？</h1>

激活函数决定非线性表达能力，归一化影响训练稳定性，损失函数决定模型优化方向。面试中通常不要求手推所有复杂导数，但要能解释关键梯度形式和数值稳定性问题。

<h2 id="q-016">面试问题：Sigmoid、Tanh、ReLU、GELU、SiLU 的导数和梯度问题是什么？</h2>

Sigmoid：

$$
\sigma(x)=\frac{1}{1+e^{-x}},\quad \sigma'(x)=\sigma(x)(1-\sigma(x))
$$

当 $x$ 很大或很小时，梯度接近 0，容易导致梯度消失。

Tanh：

$$
\tanh'(x)=1-\tanh^2(x)
$$

输出均值更接近 0，但饱和区仍有梯度消失问题。

ReLU：

$$
\text{ReLU}(x)=\max(0,x)
$$

$$
\text{ReLU}'(x)=
\begin{cases}
1, & x>0 \\
0, & x<0
\end{cases}
$$

优点是简单、缓解梯度消失；缺点是可能出现 dead ReLU。

GELU 常用于 Transformer：

$$
\text{GELU}(x)=x\Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布 CDF。它可以看作平滑的、带概率门控直觉的激活函数。

SiLU/Swish：

$$
\text{SiLU}(x)=x\sigma(x)
$$

在很多现代网络中表现稳定，LLaMA 系列常使用 SwiGLU 变体。

<h2 id="q-017">面试问题：Softmax + Cross Entropy 为什么梯度形式很简洁？</h2>

Softmax：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

交叉熵损失：

$$
L=-\sum_i y_i\log p_i
$$

当 $y$ 是 one-hot 标签时：

$$
\frac{\partial L}{\partial z_i}=p_i-y_i
$$

这说明分类任务中 logits 的梯度就是“预测概率 - 真实标签”。这也是 Softmax 和 Cross Entropy 通常合并实现的原因：梯度简单，且可以通过 log-sum-exp 技巧提升数值稳定性。

**大模型中对应形式：**

语言模型每个位置预测下一个 token：

$$
L=-\sum_{t=1}^{T}\log p_{\theta}(x_t|x_{<t})
$$

本质上就是对词表做 Softmax 分类。

<h2 id="q-018">面试问题：LayerNorm、RMSNorm 的数学作用是什么？</h2>

LayerNorm 对单个样本的 hidden 维度做归一化：

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\text{LN}(x)=\gamma \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm 去掉均值中心化，只按均方根缩放：

$$
\text{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

$$
\text{RMSNorm}(x)=\gamma \frac{x}{\text{RMS}(x)}
$$

**区别：**

- LayerNorm 同时做中心化和缩放。
- RMSNorm 只做缩放，计算更省，现代 LLM 中很常见。
- 二者都能稳定激活分布，改善梯度传播。

---

<h1 id="q-019">5. 机器学习为什么需要概率论？</h1>

机器学习处理的是不确定性：数据有噪声、标签有误差、模型参数未知、生成结果具有随机性、用户偏好不可完全观测。概率论提供了描述不确定性、定义训练目标、推导损失函数和评估模型置信度的统一语言。

<h2 id="q-020">面试问题：随机变量、概率分布、PMF、PDF 分别是什么？</h2>

随机变量是把随机试验结果映射为数值的函数。随机变量必须配合概率分布才完整。

普通变量与随机变量的区别在于取值是否确定。如果变量 $x=100$ 的概率为 1，它就是确定变量；如果 $x$ 可能取 50 或 100，且概率分别为 0.5 和 0.5，它就是随机变量。

离散随机变量用概率质量函数 PMF 描述：

$$
P(X=x)
$$

需要满足：

$$
0 \le P(X=x) \le 1,\quad \sum_x P(X=x)=1
$$

连续随机变量用概率密度函数 PDF 描述：

$$
p(x)
$$

需要满足：

$$
p(x)\ge 0,\quad \int p(x)dx=1
$$

连续变量在单点处的概率通常为 0，区间概率由积分给出：

$$
P(a \le X \le b)=\int_a^b p(x)dx
$$

<h2 id="q-021">面试问题：联合概率、边缘概率、条件概率、贝叶斯公式有什么联系？</h2>

联合概率描述多个事件同时发生：

$$
P(X=x,Y=y)
$$

边缘概率可以从联合概率求和或积分得到：

$$
P(X=x)=\sum_y P(X=x,Y=y)
$$

条件概率：

$$
P(A|B)=\frac{P(A\cap B)}{P(B)}
$$

乘法公式：

$$
P(A,B)=P(A|B)P(B)=P(B|A)P(A)
$$

贝叶斯公式：

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$

其中 $P(A)$ 是先验概率，$P(B|A)$ 是似然，$P(A|B)$ 是后验概率，$P(B)$ 是证据或归一化常数。

链式法则：

$$
P(x_1,...,x_n)=\prod_{i=1}^{n}P(x_i|x_1,...,x_{i-1})
$$

**大模型中的典型应用：**

自回归语言模型正是使用概率链式法则：

$$
P(x_1,...,x_T)=\prod_{t=1}^{T}P(x_t|x_{<t})
$$

**经典条件概率例题：**

一对夫妻有两个孩子，已知至少一个是女孩，问另一个也是女孩的概率是多少？若默认男女性别等概率且孩子有出生顺序，则样本空间为：

$$
\{GG, GB, BG, BB\}
$$

已知至少一个是女孩后排除 $BB$，剩下：

$$
\{GG, GB, BG\}
$$

因此另一个也是女孩的概率是 $1/3$。这个题容易误答为 $1/2$，原因是忽略了“至少一个是女孩”和“指定某一个孩子是女孩”的条件并不相同。

<h2 id="q-022">面试问题：独立性和条件独立性有什么区别？</h2>

若：

$$
P(X,Y)=P(X)P(Y)
$$

则 $X$ 和 $Y$ 独立。

若给定 $Z$ 后：

$$
P(X,Y|Z)=P(X|Z)P(Y|Z)
$$

则 $X$ 和 $Y$ 在 $Z$ 条件下条件独立，记作：

$$
X \perp Y | Z
$$

**注意：**

- 独立不一定推出条件独立。
- 条件独立也不一定推出无条件独立。
- 朴素贝叶斯假设“给定类别后各特征条件独立”，实际不完全成立，但常能得到可用效果。

<h2 id="q-023">面试问题：期望、方差、协方差、相关系数如何理解？</h2>

期望描述平均取值：

$$
\mathbb{E}[X]=\sum_x xP(x)
$$

连续情形：

$$
\mathbb{E}[X]=\int xp(x)dx
$$

期望具有线性性：

$$
\mathbb{E}[aX+bY+c]=a\mathbb{E}[X]+b\mathbb{E}[Y]+c
$$

方差描述随机变量围绕均值的波动：

$$
\text{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]
$$

等价形式：

$$
\text{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2
$$

协方差描述两个变量线性相关方向和强度：

$$
\text{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]
$$

相关系数是归一化协方差：

$$
\rho_{X,Y}=\frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}
$$

取值范围为 $[-1,1]$。

**面试易错点：**

- 不相关只表示线性相关为 0，不等价于独立。
- 独立通常可推出不相关，但不相关不一定推出独立。
- 协方差受量纲影响，相关系数无量纲。
- 一般情况下 $\mathbb{E}[XY] \ne \mathbb{E}[X]\mathbb{E}[Y]$，只有在 $X,Y$ 独立等条件下才成立。
- Jensen 不等式：若 $f$ 是凸函数，则 $\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])$；若 $f$ 是凹函数，方向相反。

---

<h1 id="q-024">6. 常见概率分布在 AI 中如何使用？</h1>

概率分布是机器学习建模假设的核心。分类、回归、生成、扩散、强化学习、异常检测、贝叶斯估计都需要选择合适的分布。

<h2 id="q-025">面试问题：Bernoulli、Categorical、Binomial、Multinomial 分布分别适合什么场景？</h2>

Bernoulli 分布描述单次二值试验：

$$
P(X=1)=p,\quad P(X=0)=1-p
$$

概率质量函数：

$$
P(X=x)=p^x(1-p)^{1-x}
$$

适合二分类标签、开关变量、dropout mask。

Categorical 分布描述单次多类别试验：

$$
P(X=i)=p_i,\quad \sum_i p_i=1
$$

适合多分类、token 采样。

Binomial 分布描述 $n$ 次 Bernoulli 试验中成功次数：

$$
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$

Multinomial 分布是 Binomial 的多类别推广，适合词袋模型、类别计数建模。

<h2 id="q-026">面试问题：高斯分布、多元高斯、协方差矩阵为什么重要？</h2>

一维高斯分布：

$$
\mathcal{N}(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

标准正态分布是 $\mu=0,\sigma=1$ 的特例。经验上：

- $[\mu-\sigma,\mu+\sigma]$ 约包含 68.3% 概率质量。
- $[\mu-2\sigma,\mu+2\sigma]$ 约包含 95.5% 概率质量。
- $[\mu-3\sigma,\mu+3\sigma]$ 约包含 99.7% 概率质量。

多元高斯分布：

$$
\mathcal{N}(x;\mu,\Sigma)=\frac{1}{\sqrt{(2\pi)^d|\Sigma|}}\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$

其中 $\Sigma$ 是协方差矩阵。

**为什么高斯分布常用？**

- 中心极限定理说明大量独立扰动叠加后近似高斯。
- 在固定均值和方差下，高斯分布具有最大熵，表示额外假设最少。
- VAE、扩散模型、噪声建模、初始化、误差建模都大量使用高斯分布。

<h2 id="q-027">面试问题：指数分布、Laplace 分布、Dirac 分布和经验分布有什么用途？</h2>

指数分布常用于等待时间建模：

$$
p(x;\lambda)=\lambda e^{-\lambda x},\quad x\ge 0
$$

Laplace 分布：

$$
\text{Laplace}(x;\mu,b)=\frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)
$$

它比高斯分布尾部更重，与 $L_1$ 损失有紧密联系。若假设误差服从 Laplace 分布，最大似然估计会导出 MAE 损失。

Dirac delta 分布把全部概率质量集中在一个点。经验分布可写成：

$$
\hat{p}(x)=\frac{1}{N}\sum_{i=1}^{N}\delta(x-x_i)
$$

训练集可以被看作来自真实分布的经验分布，经验风险最小化就是在经验分布上最小化平均损失。

<h2 id="q-028">面试问题：最大似然估计、MAP、贝叶斯估计有什么区别？</h2>

最大似然估计 MLE：

$$
\theta_{\text{MLE}}=\arg\max_{\theta}p(D|\theta)
$$

通常优化负对数似然：

$$
\theta_{\text{MLE}}=\arg\min_{\theta}-\sum_i \log p(x_i|\theta)
$$

最大后验估计 MAP：

$$
\theta_{\text{MAP}}=\arg\max_{\theta}p(\theta|D)=\arg\max_{\theta}p(D|\theta)p(\theta)
$$

MAP 比 MLE 多了先验 $p(\theta)$。例如高斯先验会导出 $L_2$ 正则，Laplace 先验会导出 $L_1$ 正则。

贝叶斯估计不只给出一个最优点，而是保留参数后验分布：

$$
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}
$$

**面试总结：**

- MLE：只看数据似然。
- MAP：数据似然 + 参数先验。
- 贝叶斯估计：维护完整后验，更能表达不确定性，但计算更复杂。

---

<h1 id="q-029">7. 熵、交叉熵、KL 散度、互信息为什么是大模型基础？</h1>

信息论把“预测不确定性”和“分布差异”变成可优化的数学量。语言模型训练的交叉熵、VAE 的 KL 正则、知识蒸馏的 KL 损失、RLHF 的 KL penalty、对比学习的 InfoNCE 都属于信息论在 AI 中的具体应用。

<h2 id="q-030">面试问题：熵、交叉熵和困惑度 Perplexity 如何理解？</h2>

熵衡量分布自身不确定性：

$$
H(P)=-\sum_x P(x)\log P(x)
$$

交叉熵衡量用分布 $Q$ 编码来自 $P$ 的样本时的平均编码长度：

$$
H(P,Q)=-\sum_x P(x)\log Q(x)
$$

分类任务中真实分布 $P$ 通常是 one-hot，所以交叉熵就是负对数似然。

语言模型困惑度：

$$
\text{PPL}=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_{\theta}(x_t|x_{<t})\right)
$$

困惑度越低，表示模型给真实 token 的平均概率越高。但 PPL 不是万能指标，指令跟随、事实性、推理能力、对齐质量不能只靠 PPL 判断。

<h2 id="q-031">面试问题：KL 散度为什么不对称？在 VAE、蒸馏、RLHF 中怎么用？</h2>

KL 散度定义为：

$$
D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

连续情形：

$$
D_{KL}(P\|Q)=\int p(x)\log\frac{p(x)}{q(x)}dx
$$

KL 不对称：

$$
D_{KL}(P\|Q)\ne D_{KL}(Q\|P)
$$

因为期望是在第一个分布 $P$ 上取的。若 $P$ 在某些区域概率大而 $Q$ 很小，会受到强惩罚；反过来未必。

**典型应用：**

- VAE：让近似后验 $q_\phi(z|x)$ 接近先验 $p(z)$。
- 知识蒸馏：让学生模型分布接近教师模型分布。
- RLHF/PPO：用 KL penalty 限制策略模型偏离参考模型过远。
- DPO：隐式利用 KL 约束下的偏好优化推导出分类式目标。

<h2 id="q-032">面试问题：JS 散度、互信息、最大熵原理在生成模型中有什么意义？</h2>

JS 散度是 KL 的对称平滑版本：

$$
D_{JS}(P\|Q)=\frac{1}{2}D_{KL}(P\|M)+\frac{1}{2}D_{KL}(Q\|M)
$$

其中：

$$
M=\frac{1}{2}(P+Q)
$$

GAN 原始目标与 JS 散度有关，但当真实分布和生成分布支撑集几乎不重叠时，训练可能不稳定。

互信息衡量两个随机变量共享的信息：

$$
I(X;Y)=D_{KL}(P(X,Y)\|P(X)P(Y))
$$

也可写为：

$$
I(X;Y)=H(X)-H(X|Y)
$$

**AI 中用途：**

- 对比学习希望正样本表示共享更多信息。
- 多模态学习希望图文表示对齐。
- 表征学习希望保留任务相关信息，压缩无关噪声。

最大熵原理：在已知约束下选择熵最大的分布，避免引入额外假设。Softmax、最大熵强化学习、温度采样都与这个思想相关。

---

<h1 id="q-033">8. 梯度下降与现代优化器的数学本质是什么？</h1>

优化是把数学目标变成可训练模型的桥梁。大模型训练的核心困难不仅是“有没有梯度”，还包括梯度噪声、尺度不稳定、学习率调度、混合精度数值误差、分布式训练同步和泛化。

<h2 id="q-034">面试问题：SGD、Momentum、AdaGrad、RMSProp、Adam、AdamW 有什么区别？</h2>

普通梯度下降：

$$
\theta_{t+1}=\theta_t-\eta \nabla_{\theta}L(\theta_t)
$$

SGD 使用 mini-batch 梯度近似全量梯度。

Momentum 引入速度项：

$$
v_t=\beta v_{t-1}+g_t
$$

$$
\theta_{t+1}=\theta_t-\eta v_t
$$

它能平滑梯度噪声，在一致下降方向上加速。

AdaGrad 为不同参数累计历史平方梯度：

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t+\epsilon}}g_t
$$

适合稀疏特征，但学习率会不断衰减。

RMSProp 使用指数滑动平均缓解 AdaGrad 衰减过快：

$$
s_t=\beta s_{t-1}+(1-\beta)g_t^2
$$

Adam 同时估计一阶矩和二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

带偏差修正：

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad \hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

更新：

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

AdamW 将 weight decay 与 Adam 的梯度更新解耦，是训练 Transformer 和 LLM 的常用优化器。

<h2 id="q-035">面试问题：学习率、Batch Size、梯度噪声、Warmup、Cosine Decay 如何影响训练？</h2>

学习率决定每步参数更新幅度。太大可能发散，太小训练慢或陷入平台区。

Batch size 越大，梯度估计方差越小，但泛化未必更好，且需要配合学习率调整。常见经验是大 batch 下可适当增大学习率，但需要 warmup。

Warmup 的作用：训练初期参数、激活和优化器状态还不稳定，逐步升高学习率可以避免早期大步更新导致发散。

Cosine decay：

$$
\eta_t=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

它让训练后期学习率平滑降低，有利于收敛。

**大模型训练常见组合：**

- AdamW
- Linear warmup
- Cosine decay
- Gradient clipping
- Mixed precision + loss scaling

<h2 id="q-036">面试问题：梯度爆炸、梯度消失、梯度裁剪如何从数学上理解？</h2>

深层网络反向传播会连续相乘 Jacobian：

$$
\frac{\partial L}{\partial x_0}=\frac{\partial L}{\partial x_n}\prod_{i=1}^{n}\frac{\partial x_i}{\partial x_{i-1}}
$$

若这些 Jacobian 的谱范数长期小于 1，梯度会指数衰减；若长期大于 1，梯度会指数放大。

梯度裁剪常见形式：

$$
g \leftarrow g \cdot \min\left(1,\frac{c}{\|g\|_2}\right)
$$

当梯度范数超过阈值 $c$ 时按比例缩小，方向不变，长度受控。

**大模型中的意义：**

- 防止某些 batch 或长序列导致异常大梯度破坏训练。
- 与 mixed precision 结合时，可降低数值溢出风险。
- 对 RNN、Transformer、RLHF 训练尤其常见。

<h2 id="q-037">面试问题：过拟合、正则化、Dropout、Label Smoothing 的数学作用是什么？</h2>

过拟合是模型在训练集经验风险很低，但真实分布上的泛化误差较高。

$L_2$ 正则：

$$
L' = L+\lambda\|\theta\|_2^2
$$

它限制参数过大，等价于给参数加高斯先验。

$L_1$ 正则：

$$
L'=L+\lambda\|\theta\|_1
$$

它鼓励稀疏，等价于 Laplace 先验。

Dropout 训练时随机置零部分激活，可以看作训练许多子网络的近似集成。

Label Smoothing 将 one-hot 标签：

$$
y_i \in \{0,1\}
$$

变成：

$$
y_i'=(1-\epsilon)y_i+\frac{\epsilon}{K}
$$

它避免模型过度自信，提升校准性，但在某些蒸馏或偏好优化场景可能需要谨慎使用。

<h2 id="q-037a">面试问题：无约束优化、有约束优化、拉格朗日乘子和 KKT 条件如何理解？</h2>

无约束优化形式：

$$
\min_x f(x)
$$

常见方法包括梯度下降、Momentum、Adam、Newton 法、拟 Newton 法等。深度学习训练大多数时候可看作大规模随机无约束优化。

等式约束优化形式：

$$
\min_x f(x),\quad \text{s.t. } h_i(x)=0
$$

拉格朗日函数：

$$
\mathcal{L}(x,\lambda)=f(x)+\sum_i\lambda_i h_i(x)
$$

最优点需要满足驻点条件：

$$
\nabla_x\mathcal{L}(x,\lambda)=0
$$

不等式约束优化形式：

$$
\min_x f(x),\quad \text{s.t. } g_i(x)\le 0,\ h_j(x)=0
$$

KKT 条件包括：

- 原始可行性：$g_i(x)\le 0,\ h_j(x)=0$
- 对偶可行性：$\lambda_i\ge 0$
- 互补松弛：$\lambda_i g_i(x)=0$
- 驻点条件：$\nabla_x \mathcal{L}(x,\lambda,\nu)=0$

**AI 中的典型联系：**

- SVM 的最大间隔推导依赖拉格朗日对偶和 KKT。
- PPO、DPO、RLHF 中的 KL 约束可以看作约束优化或正则化优化。
- 权重范数约束、谱范数约束、正交约束都属于约束优化思想。
- 实践中常把硬约束转成软惩罚，例如 $L+\lambda R(\theta)$。

<h2 id="q-037b">面试问题：凸优化和非凸优化有什么区别？深度学习为什么仍能训练成功？</h2>

若函数 $f$ 满足：

$$
f(\alpha x+(1-\alpha)y)\le \alpha f(x)+(1-\alpha)f(y),\quad \alpha\in[0,1]
$$

则 $f$ 是凸函数。

凸优化的优势：

- 任意局部最优都是全局最优。
- 理论分析更完整。
- 梯度下降、坐标下降、内点法等方法有较清晰的收敛性质。

深度学习优化通常是非凸的，因为多层网络、注意力、归一化、激活函数组合后形成复杂损失曲面。

**为什么非凸仍能训练成功？**

- 过参数化：参数量远大于约束数量，存在大量可行低损失解。
- SGD 噪声：mini-batch 噪声有助于逃离部分尖锐区域。
- 残差连接和归一化：改善梯度传播和损失地形。
- 好的初始化：让网络从稳定区域开始训练。
- 学习率调度：先探索后收敛。
- 实践中更关心泛化好的低损失解，而不是严格全局最优。

**面试易错点：**

非凸不等于无法优化。现代深度网络的成功依赖模型结构、数据规模、优化器、归一化、初始化和训练技巧共同作用。

---

<h1 id="q-038">9. Attention 的核心数学是什么？</h1>

Attention 的本质是基于相似度的加权求和。Query 表示“我要找什么”，Key 表示“我有什么索引”，Value 表示“真正被聚合的信息”。Transformer 用可微分的注意力机制替代固定窗口或递归结构，实现全局依赖建模。

<h2 id="q-039">面试问题：Scaled Dot-Product Attention 为什么要除以 $\sqrt{d_k}$？</h2>

注意力公式：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

若 $q$ 和 $k$ 的每个分量均值为 0、方差为 1，则点积：

$$
q \cdot k = \sum_{i=1}^{d_k}q_ik_i
$$

方差约为 $d_k$。当 $d_k$ 很大时，logits 绝对值变大，Softmax 容易进入饱和区，梯度变小。

除以 $\sqrt{d_k}$ 后，点积分数方差被缩放到稳定范围，有利于训练。

<h2 id="q-040">面试问题：Multi-Head Attention 的线性代数本质是什么？</h2>

多头注意力把 hidden space 投影到多个子空间：

$$
\text{head}_i=\text{Attention}(XW_i^Q,XW_i^K,XW_i^V)
$$

拼接后再线性变换：

$$
\text{MHA}(X)=\text{Concat}(\text{head}_1,...,\text{head}_H)W^O
$$

**直觉：**

- 不同 head 可以关注不同关系，如局部语法、长程指代、格式结构、代码依赖。
- 多头不是简单复制，而是多个学习到的投影子空间。
- 实际研究中也发现存在冗余 head，因此剪枝、GQA、MQA 可以减少推理成本。

<h2 id="q-041">面试问题：位置编码、RoPE、ALiBi 的数学差异是什么？</h2>

Transformer 自注意力本身对 token 顺序不敏感，所以需要位置编码。

绝对位置编码把位置向量加到 token embedding 上：

$$
h_i = x_i + p_i
$$

RoPE（Rotary Position Embedding）把位置信息注入到 $Q,K$ 的旋转变换中。二维子空间中可理解为：

$$
R_{\theta}(m)=
\begin{bmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{bmatrix}
$$

位置 $m$ 的向量被旋转后，$q_m^Tk_n$ 会自然包含相对位置 $m-n$ 的信息。

ALiBi 在注意力 logits 中加入与距离相关的线性偏置：

$$
\text{score}_{ij}=q_i^Tk_j + b_h(i-j)
$$

**区别：**

- 绝对位置编码简单，但长度外推较弱。
- RoPE 兼顾相对位置信息，是现代 LLM 主流方案之一。
- ALiBi 不需要位置 embedding，偏向长度外推和简洁性。

<h2 id="q-042">面试问题：KV Cache、FlashAttention、长上下文优化背后的数学直觉是什么？</h2>

自回归推理中，第 $t$ 步只会新增一个 token。历史 token 的 $K,V$ 不变，因此可以缓存：

$$
K_{1:t},V_{1:t}
$$

下一步只计算新 token 的 $Q$ 与历史 $K$ 的注意力，避免重复计算。

FlashAttention 不改变注意力数学公式，而是重排计算方式，分块计算 Softmax，减少 HBM 显存读写，并通过 online softmax 保持数值精确。

长上下文优化常见方向：

- 稀疏注意力：只计算部分 token 间注意力，降低 $S^2$ 成本。
- Sliding window attention：局部窗口内注意。
- GQA/MQA：减少 KV head 数量，降低 KV cache 显存。
- RoPE scaling：调整旋转频率，改善长度外推。
- 分块和检索：把长上下文问题转化为分块记忆或外部向量检索问题。

---

<h1 id="q-043">10. 大模型预训练、微调和推理中的常考数学是什么？</h1>

LLM 的核心目标是对 token 序列建模。预训练阶段学习通用概率分布，指令微调阶段改变条件分布，对齐阶段让输出分布更符合人类偏好，推理阶段通过采样策略从分布中生成文本。

<h2 id="q-044">面试问题：自回归语言模型的最大似然训练目标是什么？</h2>

给定序列 $x=(x_1,...,x_T)$，自回归分解：

$$
p_{\theta}(x)=\prod_{t=1}^{T}p_{\theta}(x_t|x_{<t})
$$

最大似然训练等价于最小化负对数似然：

$$
L(\theta)=-\sum_{t=1}^{T}\log p_{\theta}(x_t|x_{<t})
$$

对 batch 求平均后就是常见 Cross Entropy Loss。

**Causal mask 的作用：**

训练时每个位置只能看见当前位置之前的 token，避免信息泄露：

$$
\text{mask}_{ij}=
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

将 mask 加到 attention logits 后，未来 token 的注意力概率变为 0。

<h2 id="q-045">面试问题：Temperature、Top-k、Top-p、重复惩罚如何改变采样分布？</h2>

Temperature 调整 logits：

$$
p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

- $T<1$：分布更尖锐，输出更确定。
- $T>1$：分布更平坦，输出更多样。

Top-k：只保留概率最高的 $k$ 个 token，再归一化。

Top-p/Nucleus sampling：选择累计概率达到 $p$ 的最小 token 集合，再归一化。

重复惩罚通常对已经出现过的 token logits 做惩罚，降低重复生成概率。

**面试要点：**

- 解码策略不改变模型参数，只改变从条件分布中抽样的方式。
- 贪心和 beam search 未必最适合开放式生成，可能降低多样性。
- 推理任务更偏确定性，创意写作更偏多样性。

<h2 id="q-046">面试问题：Embedding 相似度、余弦相似度、向量检索和 RAG 的数学基础是什么？</h2>

Embedding 把离散对象映射到连续向量空间：

$$
e = f_{\theta}(x) \in \mathbb{R}^{d}
$$

余弦相似度：

$$
\cos(a,b)=\frac{a^Tb}{\|a\|_2\|b\|_2}
$$

若向量已归一化，内积等于余弦相似度。

RAG 检索通常包括：

1. 将 query 和文档 chunk 编码为向量。
2. 使用内积、余弦或 L2 距离找近邻。
3. 把检索结果拼接进上下文，条件生成答案。

**常见指标：**

- Recall@k：相关文档是否进入前 k。
- MRR：第一个相关结果排名的倒数均值。
- nDCG：考虑相关性等级和排序位置。

<h2 id="q-047">面试问题：MoE 的门控函数、负载均衡损失和稀疏激活如何理解？</h2>

Mixture of Experts（MoE）使用门控网络为 token 选择专家：

$$
g(x)=\text{softmax}(W_gx)
$$

Top-k gating 只激活得分最高的 $k$ 个专家：

$$
y=\sum_{i \in \text{TopK}(g(x))}g_i(x)E_i(x)
$$

**优势：**

- 总参数量很大，但每个 token 只激活部分专家。
- 计算量近似由激活专家数决定，而非总专家数决定。

**负载均衡损失：**

如果所有 token 都路由到少数专家，会造成训练不稳定和硬件负载不均。负载均衡损失鼓励专家使用更均匀。

**常见追问：MoE 是不是一定更强？**

不一定。MoE 带来路由、通信、负载均衡、专家塌缩、推理部署复杂性，需要足够规模和良好工程支持才稳定受益。

---

<h1 id="q-048">11. 扩散模型的概率建模基础是什么？</h1>

扩散模型、SDE、ODE、Flow Matching、Rectified Flow 是 AIGC 图像和多模态生成的重要数学基础。它们的共同目标是学习从简单分布到数据分布的可控变换。

<h2 id="q-049">面试问题：DDPM 的前向扩散和反向去噪如何用马尔可夫链表示？</h2>

DDPM 前向过程逐步加高斯噪声：

$$
q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)
$$

定义：

$$
\alpha_t=1-\beta_t,\quad \bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

可以一步采样任意时刻：

$$
q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

重参数化：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

反向过程学习：

$$
p_{\theta}(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_{\theta}(x_t,t),\sigma_t^2I)
$$

常见训练目标是预测噪声：

$$
L_{\text{simple}}=\mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon-\epsilon_{\theta}(x_t,t)\|_2^2\right]
$$

**面试要点：**

- 前向扩散是固定马尔可夫链。
- 反向去噪是可学习马尔可夫链。
- 训练时随机采样时间步，无需从 1 到 $t$ 逐步加噪。

<h2 id="q-050">面试问题：Score Matching、SDE、ODE 与扩散模型有什么关系？</h2>

Score 是数据对数密度对样本的梯度：

$$
\nabla_x \log p_t(x)
$$

它指向概率密度上升最快方向。Score-based model 学习不同噪声尺度下的 score，用它从噪声逐步回到数据分布。

连续时间扩散可写为 SDE：

$$
dx=f(x,t)dt+g(t)dw
$$

反向时间 SDE：

$$
dx=[f(x,t)-g(t)^2\nabla_x\log p_t(x)]dt+g(t)d\bar{w}
$$

对应还存在 Probability Flow ODE：

$$
dx=\left[f(x,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\right]dt
$$

**区别：**

- SDE 采样带随机性。
- ODE 采样确定性，更适合反演、编辑、似然计算。
- DDPM 可看作离散时间扩散，SDE 是连续时间统一框架。

<h2 id="q-051">面试问题：Classifier-Free Guidance 的数学形式是什么？为什么 CFG scale 过大会失真？</h2>

Classifier-Free Guidance 同时训练条件和无条件预测。以噪声预测为例：

$$
\hat{\epsilon}=\epsilon_{\theta}(x_t,\varnothing,t)+s\left(\epsilon_{\theta}(x_t,c,t)-\epsilon_{\theta}(x_t,\varnothing,t)\right)
$$

其中 $s$ 是 guidance scale。

直觉：无条件分支给出“自然图像先验”，条件分支给出“满足文本条件的方向”，二者差值代表条件引导方向。

当 $s$ 过大时：

- 模型过度追逐条件方向。
- 样本可能偏离真实数据流形。
- 容易出现过饱和、纹理失真、构图僵硬、多样性下降。

工程上常通过动态 CFG、阈值裁剪、降低 scale、改进 prompt 或使用更强文本图像对齐模型缓解。

<h2 id="q-052">面试问题：Flow Matching、Rectified Flow 与传统扩散模型的训练目标有什么差异？</h2>

Flow Matching 直接学习一个连续速度场 $v_{\theta}(x_t,t)$，把简单分布 $p_0$ 传输到数据分布 $p_1$。

若采用线性插值路径：

$$
x_t=(1-t)x_0+tx_1
$$

目标速度为：

$$
u_t=x_1-x_0
$$

训练目标：

$$
L=\mathbb{E}_{t,x_t}\left[\|v_{\theta}(x_t,t)-u_t(x_t)\|_2^2\right]
$$

Rectified Flow 也学习从噪声到数据的速度场，强调把生成路径“拉直”，使少步采样更有效。

**与 DDPM 的差异：**

- DDPM 常学习噪声、score 或 $x_0$ 预测。
- Flow Matching 学习速度场。
- DDPM 采样通常是逐步去噪；Flow/Rectified Flow 更像求解 ODE 轨迹。
- 现代图像生成模型中，Flow Matching/Rectified Flow 因少步采样效率和稳定性受到重视。

---

<h1 id="q-053">12. 大模型对齐为什么需要偏好优化和强化学习？</h1>

预训练让模型学会“像互联网文本一样续写”，但不保证它有帮助、诚实、安全、符合用户偏好。对齐学习把人类偏好、规则约束和任务反馈转化为优化目标，让模型输出更符合实际需求。

<h2 id="q-054">面试问题：RLHF 中 reward model、PPO、KL penalty 分别解决什么问题？</h2>

RLHF 通常包括：

1. SFT：用高质量指令数据做监督微调。
2. Reward Model：根据人类偏好比较学习奖励函数。
3. RL 优化：用 PPO 等方法最大化奖励，同时限制偏离参考模型。

奖励模型常用偏好对 $(y_w,y_l)$ 训练：

$$
L_{\text{RM}}=-\log\sigma(r_{\phi}(x,y_w)-r_{\phi}(x,y_l))
$$

PPO 优化策略时使用 clipped objective 控制更新幅度。

KL penalty 限制策略模型 $\pi_{\theta}$ 不要偏离参考模型 $\pi_{\text{ref}}$ 太远：

$$
R'(x,y)=R(x,y)-\beta D_{KL}(\pi_{\theta}(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))
$$

**为什么需要 KL？**

只最大化 reward 容易 reward hacking，模型可能找到奖励模型漏洞，输出怪异或不自然内容。KL 约束保留语言模型先验。

<h2 id="q-055">面试问题：DPO 如何把偏好学习转化为分类式损失？</h2>

DPO（Direct Preference Optimization）不显式训练 reward model，也不跑 PPO，而是从 KL 约束下的偏好优化推导出直接损失。

给定 prompt $x$，偏好回答 $y_w$，劣选回答 $y_l$，DPO 损失：

$$
L_{\text{DPO}}=-\log\sigma\left(\beta\left[\log\frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\log\frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)
$$

直觉：

- 提高模型相对参考模型生成偏好回答的概率。
- 降低模型相对参考模型生成劣选回答的概率。
- $\beta$ 控制偏离参考模型的强度。

**优势：**

- 训练流程简单稳定。
- 不需要在线 RL 采样。
- 适合偏好数据充足的对齐场景。

<h2 id="q-056">面试问题：GRPO 与 PPO 的数学差异是什么？为什么适合推理模型训练？</h2>

GRPO（Group Relative Policy Optimization）使用同一 prompt 下的一组候选回答作为比较组，用组内相对奖励估计 advantage，减少对单独 value model 的依赖。

对同一问题采样多个回答：

$$
\{y_1,...,y_G\}
$$

得到奖励：

$$
\{r_1,...,r_G\}
$$

组内标准化 advantage：

$$
A_i=\frac{r_i-\text{mean}(r_1,...,r_G)}{\text{std}(r_1,...,r_G)}
$$

再用类似 PPO 的 clipped ratio 目标更新策略，并加入 KL 约束。

**为什么适合推理模型？**

- 数学、代码、逻辑推理任务常能自动判分或弱监督判分。
- 同一题多条推理路径可比较，组内相对优势更自然。
- 省去 value model，降低训练复杂度。

---

<h1 id="q-057">13. LoRA、QLoRA、量化和蒸馏背后的数学是什么？</h1>

参数高效微调和部署优化是 AIGC 落地必备能力。它们的共同目标是：用更少可训练参数、更低显存、更低计算成本，尽量保持模型能力。

<h2 id="q-058">面试问题：LoRA 为什么可以用低秩矩阵近似权重更新？</h2>

原始线性层：

$$
y=Wx
$$

微调时不直接更新 $W$，而是学习低秩增量：

$$
W'=W+\Delta W,\quad \Delta W=BA
$$

其中：

$$
B \in \mathbb{R}^{d_{out}\times r},\quad A \in \mathbb{R}^{r\times d_{in}},\quad r \ll \min(d_{out},d_{in})
$$

前向：

$$
y=Wx+\frac{\alpha}{r}BAx
$$

**数学假设：**

下游任务所需的参数更新 $\Delta W$ 近似低秩，即主要变化集中在少数方向上。

**优势：**

- 可训练参数量从 $d_{out}d_{in}$ 降到 $r(d_{out}+d_{in})$。
- 原模型权重冻结，便于多任务切换。
- 推理时可将 LoRA 权重 merge 回原权重。

<h2 id="q-059">面试问题：量化、反量化、对称/非对称量化、NF4 的核心数学是什么？</h2>

量化把高精度浮点数映射为低比特整数或离散码本值。

对称量化：

$$
q=\text{round}\left(\frac{x}{s}\right)
$$

$$
\hat{x}=s q
$$

其中 $s$ 是 scale。

非对称量化加入 zero point：

$$
q=\text{round}\left(\frac{x}{s}+z\right)
$$

$$
\hat{x}=s(q-z)
$$

量化误差：

$$
e=x-\hat{x}
$$

**常见粒度：**

- per-tensor：整个张量一个 scale，简单但误差大。
- per-channel：每个通道一个 scale，更精确。
- group-wise：按组量化，是 LLM 权重量化常见折中。

NF4 是 QLoRA 中常用的 4-bit NormalFloat 格式，针对近似正态分布的权重设计非均匀量化码本，相比均匀 int4 更适合神经网络权重分布。

<h2 id="q-060">面试问题：知识蒸馏、温度系数和 KL 损失如何理解？</h2>

知识蒸馏让学生模型学习教师模型输出分布，而不只是学习 hard label。

带温度的 Softmax：

$$
p_i^{(T)}=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

蒸馏损失常用 KL：

$$
L_{\text{KD}}=T^2D_{KL}(p_T^{\text{teacher}}\|p_T^{\text{student}})
$$

温度 $T$ 越大，分布越平滑，学生能学习到类别之间的“暗知识”。在 LLM 中，蒸馏还可以表现为：

- logits 蒸馏
- chain-of-thought 蒸馏
- preference distillation
- step-by-step reasoning distillation

---

<h1 id="q-061">14. AIGC 数学基础应该如何复习？</h1>

AIGC 时代的数学复习不能只按传统教材章节堆知识，更应该围绕“哪些数学对象会直接进入模型结构、损失函数、训练算法和推理部署”来组织。

<h2 id="q-062">面试问题：算法岗、大模型工程岗、多模态方向各自最该优先掌握哪些数学？</h2>

**通用必备：**

- 线性代数：矩阵乘法、范数、特征值、SVD、正定矩阵、低秩近似。
- 微积分：偏导、梯度、链式法则、Jacobian、Hessian、常见损失求导。
- 概率统计：随机变量、常见分布、条件概率、贝叶斯、MLE/MAP、期望方差。
- 信息论：熵、交叉熵、KL、互信息、困惑度。
- 优化：SGD/AdamW、学习率调度、正则化、梯度裁剪、训练稳定性。

**大语言模型方向：**

- 自回归分解和最大似然。
- Attention 复杂度与 $\sqrt{d_k}$ 缩放。
- 位置编码、RoPE、KV Cache、FlashAttention。
- RLHF、DPO、GRPO、KL 约束。
- LoRA、量化、蒸馏、MoE。

**多模态与图像生成方向：**

- 高斯噪声、马尔可夫链、重参数化。
- DDPM、Score Matching、SDE/ODE。
- CFG、Flow Matching、Rectified Flow。
- VAE 的 ELBO 和 KL 正则。
- CLIP/对比学习、图文相似度、跨模态检索。

**工程落地方向：**

- 张量 shape 推导。
- 显存与复杂度估算。
- 混合精度和数值稳定。
- 向量检索指标。
- 量化误差与吞吐/延迟权衡。

**一句话总结：**

面试里最好的数学回答不是只背公式，而是能说清楚：这个公式在模型里对应哪个模块、解决什么训练或推理问题、有什么工程代价，以及参数变化会带来什么现象。

---

## 参考资料

[1] Ian Goodfellow, Yoshua Bengio, Aaron Courville. *Deep Learning*. MIT Press, 2016.

[2] 周志华. 《机器学习》. 清华大学出版社, 2016.

[3] Vaswani et al. *Attention Is All You Need*. NeurIPS, 2017. <https://arxiv.org/abs/1706.03762>

[4] Ho et al. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020. <https://arxiv.org/abs/2006.11239>

[5] Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR, 2021. <https://arxiv.org/abs/2011.13456>

[6] Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR, 2022. <https://arxiv.org/abs/2106.09685>

[7] Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS, 2023. <https://arxiv.org/abs/2305.14314>

[8] Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS, 2023. <https://arxiv.org/abs/2305.18290>

[9] Lipman et al. *Flow Matching for Generative Modeling*. ICLR, 2023. <https://arxiv.org/abs/2210.02747>

[10] Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS, 2022. <https://arxiv.org/abs/2205.14135>

[11] Fedus et al. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR, 2022. <https://arxiv.org/abs/2101.03961>

[12] DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. 2025. <https://arxiv.org/abs/2501.12948>
