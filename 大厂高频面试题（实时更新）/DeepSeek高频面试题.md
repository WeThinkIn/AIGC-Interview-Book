# DeepSeek高频面试题

> 用于持续沉淀 DeepSeek、推理模型、MoE、强化学习、代码/数学能力、训练与推理优化相关面试题。

## 待补充方向

本文件作为该公司面试题的持续更新入口。后续可按照“岗位/面试轮次/问题/答案/手撕代码/复盘要点”的格式补充内容。

建议优先补充以下方向：

- 大模型、AIGC、多模态、Agent 或推荐搜索相关岗位面经。
- 公司核心业务场景中的算法工程问题。
- 高频手撕代码、系统设计、训练/推理/部署工程题。

## 目录
- [1. DeepSeek 大模型算法岗笔试面经（2026-07-26）](#deepseek-llm-algo-20260726)
  - [1. 题目一：手写完整的Multi-Head Attention，不能只写框架](#deepseek-llm-algo-20260726-q1)
  - [2. 题目二：DPO的完整训练流程推导，从数据准备到梯度更新](#deepseek-llm-algo-20260726-q2)
  - [3. 题目三：MOE模型的通信开销计算和负载不均衡问题分析](#deepseek-llm-algo-20260726-q3)
  - [4. 题目四：推理加速的底层实现（vLLM的PagedAttention原理、投机解码的工程实现）](#deepseek-llm-algo-20260726-q4)
  - [5. 题目五：DSpark推理加速框架的核心机制推导](#deepseek-llm-algo-20260726-q5)
  - [6. 题目六：DeepSeek-V4的KV缓存压缩原理与内存占用计算](#deepseek-llm-algo-20260726-q6)
  - [7. 题目七：多模态视觉原语推理框架的数学建模](#deepseek-llm-algo-20260726-q7)
  - [8. 题目八：DeepSeek MoE的无辅助损失负载均衡机制推导](#deepseek-llm-algo-20260726-q8)

<a id="deepseek-llm-algo-20260726"></a>
### 1. DeepSeek 大模型算法岗笔试面经（2026-07-26）

#### 面试问题汇总

<a id="deepseek-llm-algo-20260726-q1"></a>
##### 1. 题目一：手写完整的Multi-Head Attention，不能只写框架

**回答：**

这道题不能只写四个线性层，完整回答至少要把张量形状、缩放点积、掩码语义、数值稳定性和输出投影讲清楚。

设输入为 $X\in\mathbb{R}^{B\times S\times d_{model}}$，头数为 $H$，则每个头的维度为 $d_h=d_{model}/H$。先做三组线性投影：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

其中 $Q,K,V$ 的形状都是 $[B,S,d_{model}]$。将最后一维拆成多头并交换维度后，形状变为 $[B,H,S,d_h]$。每个头的注意力为：

$$
A=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}}+M\right),\qquad
O=AV
$$

最后把 $H$ 个头拼回 $[B,S,d_{model}]$，再经过 $W_O$ 得到输出。除以 $\sqrt{d_h}$ 是为了控制点积方差，避免 logits 随维度增大而使 Softmax 饱和、梯度变小。

一个可以现场运行的简化实现如下。这里约定布尔 `attn_mask=True` 表示允许关注，形状可以是 `[B, 1, Q, K]` 或可广播到该形状的张量：

```python
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, is_causal=False):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, S, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if is_causal:
            causal = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)
```

需要主动说明几个边界。第一，`view` 只能在内存布局满足条件时安全使用，合并转置后的头通常要先 `contiguous()`；第二，因果 Mask 与 Padding Mask 的形状和广播规则不能混为一谈；第三，Cross-Attention 时 $Q$ 的长度可以与 $K,V$ 不同，不能把所有序列长度写死为 $S$。在现代大模型中还可能使用 GQA/MQA、RoPE、FlashAttention 或 fused kernel，但这些是对 QKV 组织和内核的优化，不改变上述注意力的数学骨架。

<a id="deepseek-llm-algo-20260726-q2"></a>
##### 2. 题目二：DPO的完整训练流程推导，从数据准备到梯度更新

**回答：**

DPO 的关键不是“把语言模型直接当成奖励模型”，而是从 KL 约束下的最优策略形式中消去显式奖励模型，得到只依赖偏好对数概率的目标。

训练数据是一组三元组 $(x,y_w,y_l)$：$x$ 是提示，$y_w$ 是偏好回答，$y_l$ 是非偏好回答。参考模型 $\pi_{ref}$ 通常是冻结的 SFT 模型，待训练策略是 $\pi_\theta$。

从 KL 正则化的 RLHF 目标出发：

$$
\max_\pi\;\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]
-\beta D_{KL}\left(\pi(\cdot|x)\middle\|\pi_{ref}(\cdot|x)\right)
$$

其最优策略满足：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

于是隐式奖励可以写成：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)
$$

偏好数据常用 Bradley-Terry 模型：

$$
P(y_w\succ y_l|x)=\sigma\left(r(x,y_w)-r(x,y_l)\right)
$$

两个回答的 $\beta\log Z(x)$ 会相互抵消，把 $\pi^*$ 用待训练策略近似，就得到 DPO 损失：

$$
\mathcal{L}_{DPO}(\theta)=-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right)
$$

其中序列对数概率是回答 token 条件概率之和：

$$
\log\pi(y|x)=\sum_{t\in\text{completion}}\log\pi(y_t|x,y_{<t})
$$

实际训练流程如下：

1. 清洗偏好对，确认 chosen/rejected 的 prompt 一致，处理重复、长度异常和标签噪声。
2. 用策略模型和冻结参考模型分别对 chosen、rejected 做 teacher forcing，只对 completion 部分累加 log-prob，不能把 prompt 和 padding 算进去。
3. 按上式计算 log-ratio、margin 和损失；只对策略模型反向传播，参考模型不更新。
4. 使用 AdamW、学习率和 $\beta$ 等超参数更新策略模型；实践中常从 SFT 模型开始，并监控 KL 漂移、偏好准确率、长度偏差和通用能力。
5. 用独立偏好集、事实性集和安全集评测，不能只看训练损失。

令括号内的量为 $z$，梯度方向可写为 $-\beta\sigma(-z)\nabla z$：当策略模型还没有把 chosen 与 rejected 拉开时，梯度大；margin 已经很大时，样本梯度自然变小。DPO 不需要在线 rollout、单独 Reward Model 或 PPO，但它依然依赖高质量偏好数据和合理参考模型。长回答的 log-prob 还可能带来长度偏置，因此要关注长度归一化、数据配比以及 IPO、KTO、ORPO 等相邻方法的适用边界，不能把 DPO 当成对所有偏好学习问题的唯一答案。

<a id="deepseek-llm-algo-20260726-q3"></a>
##### 3. 题目三：MOE模型的通信开销计算和负载不均衡问题分析

**回答：**

MoE 的计算量与通信量要分开分析。设一个并行批次有 $T$ 个 token，隐藏维度为 $d$，每个 token 选择 $k$ 个专家，专家数为 $E$，专家并行组大小为 $P$，每个激活值使用 $b$ 字节。

路由器先为每个 token 计算专家分数并选出 Top-$k$。在 Expert Parallel 中，token 的激活需要通过 All-to-All Dispatch 发到拥有目标专家的设备；专家计算后再通过 All-to-All Combine 将结果送回原 token 所在设备。若负载均匀、每台设备本地命中比例约为 $1/P$，网络上的有效激活载荷可近似写为：

$$
V_{one-way}\approx Tkd b\left(1-\frac{1}{P}\right)
$$

两次交换的总载荷约为：

$$
V_{comm}\approx 2Tkd b\left(1-\frac{1}{P}\right)
$$

这是粗略的聚合估计；实际还要加上路由索引、padding、元数据、协议头以及拓扑带来的非均匀开销。用延迟带宽模型表达单次通信时间：

$$
T_{comm}\approx \alpha N_{collective}+\frac{V_{comm}}{BW_{eff}}
$$

其中 $\alpha$ 是集体通信启动延迟，$BW_{eff}$ 是受 PCIe、NVLink、RoCE、拥塞和消息大小共同影响的有效带宽。小 batch 时延迟项突出，大 batch 时带宽和负载倾斜更关键。

计算侧，每个专家接收的 token 数约为 $Tk/E$，若专家是标准两层 FFN，单 token 的矩阵乘开销大致与 $d\times d_{ff}$ 成正比；因此 MoE 通过只激活少数专家降低每 token 的计算量，但并不免费：路由、All-to-All、专家权重驻留和负载倾斜会吞掉收益。比如总参数很大而激活参数较小，只能说明稀疏计算特征，不能直接推出端到端吞吐。

负载不均衡要同时看三件事：专家接收 token 数的均值和最大值、负载变异系数 `std(load)/mean(load)`，以及因为 capacity 溢出被丢弃或重路由的 token 比例。常见 capacity 为：

$$
C=\left\lceil \frac{c\,Tk}{E}\right\rceil
$$

其中 $c$ 是 capacity factor。$c$ 太小会丢 token，太大则浪费显存和通信；实际需按序列长度、batch、Top-$k$ 和目标丢弃率调参。

造成倾斜的原因包括路由器偏置、热门 token、领域分布变化、专家容量不足和跨节点拓扑差异。可用辅助负载均衡损失、专家偏置控制、容量约束、随机路由、动态 batching 以及拓扑感知放置缓解。回答时还要区分“训练时路由均衡”和“推理时服务均衡”：后者除了 token 数，还要考虑专家所在设备的真实排队时间和通信路径。

<a id="deepseek-llm-algo-20260726-q4"></a>
##### 4. 题目四：推理加速的底层实现（vLLM的PagedAttention原理、投机解码的工程实现）

**回答：**

这道题应拆成两部分：PagedAttention 解决 KV Cache 的内存管理和批处理问题；投机解码通过较便宜的草稿预测减少目标模型的串行迭代次数。两者可以结合，但不是同一个算法。

### PagedAttention

标准自回归解码中，每条请求的 KV Cache 随序列长度增长。如果为每条请求预留一段连续显存，长短请求交错会产生内部碎片，动态扩容和请求结束后的回收也很困难。PagedAttention 把 KV Cache 切成固定大小的 block：

- 逻辑序列按 token 位置映射到逻辑 block；
- block table 将逻辑 block 映射到不连续的物理显存 block；
- 新 token 到来时只分配需要的 page，结束时按引用计数回收；
- 共享前缀可以让多个请求指向相同物理 block，写入时再做 Copy-on-Write。

Attention 内核按照 block table 分页读取 K、V，并完成分块归约。这样做的本质是把“连续大数组”变成操作系统式的分页地址空间，显著降低碎片、提高并发请求的显存利用率，并支持 continuous batching 和 prefix caching。它不改变注意力的理论复杂度，也不会凭空减少每个 token 的 KV 字节数；速度收益来自更高的可服务并发、较少的内存浪费和更好的调度。

### 投机解码

给定已经确认的前缀，草稿模型先自回归生成 $\gamma$ 个候选 token；目标模型随后一次前向并行验证这些位置。贪心解码可以逐位置接受与目标模型一致的前缀；随机采样若要严格保持目标分布，需要使用基于目标分布与草稿分布的 rejection sampling，并在拒绝位置进行修正采样。若所有候选都接受，还可以额外接收目标模型给出的下一个 token。

设每个候选位置在此前位置均被接受的条件下的生存概率为 $p_i$，则期望接受长度可近似写为：

$$
\mathbb{E}[L]\approx\sum_{j=1}^{\gamma}\prod_{i=1}^{j}p_i
$$

一次迭代的收益不是只看 $\mathbb{E}[L]$，还要除以草稿生成、目标验证、KV 读写和调度的总时间：

$$
\text{speedup}\approx
\frac{\text{baseline target steps time}}
{T_{draft}(\gamma)+T_{verify}(\gamma)+T_{schedule}}
\times (1+\mathbb{E}[L])
$$

这是直觉化表达，真正评测应直接测端到端 ITL、TPOT、吞吐和请求尾延迟。

工程实现的难点包括：草稿模型与目标模型 tokenizer/位置编码兼容；验证阶段正确复用和回滚 paged KV；接受长度变化时避免无效显存写入；连续 batching 下不同请求的候选长度不同；低接受率时及时退化为普通解码；以及在量化、张量并行和 prefix cache 下保持数值与调度正确。现代实现还可能使用 n-gram、EAGLE、Medusa 或半自回归草稿器，但核心判断始终是“草稿成本是否小于它节省的目标模型串行步数”。

<a id="deepseek-llm-algo-20260726-q5"></a>
##### 5. 题目五：DSpark推理加速框架的核心机制推导

**回答：**

DSpark 这类方法可以按“更强的草稿器 + 接受率感知的验证调度”来理解。题目中给出的版本名和基准数字应以对应正式论文、代码和硬件配置为准；面试回答的重点是把机制和可计算的收益说清楚。

传统自回归草稿器逐 token 生成，候选长度为 $\gamma$ 时草稿成本近似随 $\gamma$ 增长；完全并行的草稿器虽然一次预测多个位置，却缺少块内前缀依赖，后面位置的接受率容易下降。半自回归设计通常将两者折中：并行主干先产生各位置的候选表示，再由轻量顺序模块注入前缀依赖。顺序模块可以是只依赖前一位置的 Markov head，也可以通过循环状态累积更长前缀信息。

对每个候选位置，草稿器不仅输出 token，还输出一个接受置信度 $p_i$。它应表示在前面候选已经被目标模型接受的条件下，该位置继续被接受的概率，而不是未经校准的 softmax 最大值。可以用独立验证集做逐位置温度缩放或其他校准，使预测置信度与经验接受率一致。

若一次最多验证 $\gamma$ 个 token，接受长度的近似期望为：

$$
\mathbb{E}[L(\gamma)]\approx\sum_{j=1}^{\gamma}\prod_{i=1}^{j}p_i
$$

在并发服务中，验证长度还受目标模型 batch、KV Cache、SM 利用率和通信路径影响。因此调度器不是对每个请求独立选择一个固定 $\gamma$，而是根据当前 batch 的置信度序列和实测吞吐曲线，近似求解：

$$
\max_{\gamma_1,\ldots,\gamma_N}
\frac{\sum_{r=1}^{N}\left(1+\mathbb{E}[L_r(\gamma_r)]\right)}
{T_{verify}(\gamma_1,\ldots,\gamma_N)+T_{draft}+T_{overhead}}
$$

因此“置信度调度”真正优化的是单位时间确认 token 数，而不是盲目把候选块做长。结构化代码和数学文本往往具有更强的局部可预测性，接受率可能高于开放式对话，但这只是数据分布现象，不能当成所有任务的保证。

要讲清正确性边界：只要最终由目标模型验证，并在随机采样场景使用正确的拒绝采样修正，草稿器可以是近似的，目标分布仍可保持；置信度调度只影响效率，不应改变目标模型的输出分布。工程上还要做接受长度、草稿开销、ITL、吞吐、P99 延迟、显存和不同并发度的联合评测。只有在这些指标上稳定胜出，才算真正的推理加速，而不是单用户生成速度的局部提升。

<a id="deepseek-llm-algo-20260726-q6"></a>
##### 6. 题目六：DeepSeek-V4的KV缓存压缩原理与内存占用计算

**回答：**

KV Cache 的内存占用首先由层数、KV 头数、头维度、序列长度、批量和精度决定，与模型总参数量不是一回事。

标准 MHA 中，若 decoder 层数为 $L$，每层 KV 头数为 $H_{kv}$，每头维度为 $d_h$，上下文长度为 $T$，batch 为 $B$，每个元素占 $b$ 字节，则：

$$
M_{KV}=B\times T\times L\times H_{kv}\times d_h\times 2\times b
$$

最后的 $2$ 对应 K 和 V。GQA/MQA 通过降低 $H_{kv}$ 减少 Cache；量化通过降低 $b$ 减少字节数，但会引入误差。

以压缩 latent KV 为例，如果每个 token 每层只缓存一个维度为 $d_c$ 的共享 latent，并额外保留维度为 $d_r$ 的解耦位置编码 Key，则可近似写为：

$$
M_{compressed}\approx B\times T\times L\times(d_c+d_r)\times b
$$

解码时由 latent 经过投影恢复部分 K/V，计算换来的收益是显存和带宽下降，代价是额外矩阵乘、访问模式变化、量化/重建误差和实现复杂度。若使用分组量化，还要额外计入 scale、zero-point 等元数据；若采用分层或混合精度，不能用一个统一压缩比代替真实的逐层统计。

举例说，若某配置为 $L=80$、$H_{kv}=8$、$d_h=128$、$T=10^6$、FP16、batch 为 1，则标准 MHA 的 KV 字节数为：

$$
80\times10^6\times8\times128\times2\times2
=327.68\times10^9\text{ bytes}
\approx305.2\text{ GiB}
$$

这个例子只用于说明量级，不能套用到其他模型。要从一个宣称的 GB 数字反推压缩率，必须知道实际层数、KV 维度、缓存精度、是否分页、是否把临时 workspace 和权重算入显存，以及 GB 还是 GiB。仅凭“模型有多少参数”无法推出 KV Cache 大小，也不能据此证明某种压缩结构。

所以这道题的严谨答法是：先写标准公式，再写 MLA/GQA/量化等压缩如何改变公式，最后从显存、带宽、重建 FLOPs 和质量损失做联合预算。任何具体版本号、参数规模或单卡数字，都应以公开模型配置和可复现实验为准，不能把未经配置支撑的数字当作理论推导结论。

<a id="deepseek-llm-algo-20260726-q7"></a>
##### 7. 题目七：多模态视觉原语推理框架的数学建模

**回答：**

视觉原语的核心是把空间参照从模糊的自然语言描述提升为模型可以生成、比较和验证的几何对象。可以把文本 token 与视觉原语统一成一个混合序列建模问题。

给定图像 $I$、用户文本 $x$，定义视觉原语集合：

$$
\mathcal{P}=\{\text{point}(u,v),\text{box}(u_1,v_1,u_2,v_2),
\text{mask},\text{polygon},\ldots\},
$$

其中坐标通常归一化到 $[0,1]$，并约束 $u_1\le u_2$、$v_1\le v_2$。令输出空间为文本 token 空间 $\mathcal{V}$ 与原语空间 $\mathcal{P}$ 的并集，模型生成混合序列 $z=(z_1,\ldots,z_n)$：

$$
p(z|I,x)=\prod_{t=1}^{n}p(z_t|I,x,z_{<t})
$$

实际实现可以将原语离散化为特殊 token 加坐标 bin，例如 `[BOX, x1, y1, x2, y2]`；也可以让模型输出连续坐标分布或调用专门的 grounding head。一个通用的训练目标可以写成：

$$
\mathcal{L}=\mathcal{L}_{text}
+\lambda_{coord}\mathcal{L}_{coord}
+\lambda_{geom}\mathcal{L}_{geom}
+\lambda_{ground}\mathcal{L}_{ground}
$$

其中文本部分使用交叉熵，点/框可用 L1、Smooth L1、IoU/GIoU 损失，grounding 部分约束生成的原语确实指向图像中的目标。若是计数或空间关系任务，还需要任务损失与几何一致性约束。

视觉原语的价值在于让“左上角的红色物体”变成可计算的引用：模型可以生成一个框，再对框内目标分类；也可以比较两个框的中心坐标、IoU、包含关系和相对方向。这比完全依赖语言中的“那个”“附近”“左边”更容易验证。

面试中还要说清三个难点。第一，坐标误差会传播到后续引用，因此需要坐标量化校准、IoU 阈值和不确定性表示；第二，文本 token 与几何 token 的概率空间和损失尺度不同，$\lambda$ 需要通过验证集校准；第三，视觉原语必须有语法约束和合法性校验，否则可能生成越界、反向框或无法落到图像的坐标。

因此，视觉原语不是简单给多模态模型加几个坐标 token，而是把感知、指代、空间关系和生成动作放进一个可执行的中间表示。它是否带来领先结果，要看数据集、标注质量、坐标编码、grounding 评测和与纯文本 CoT 的公平对比，不能只凭“引入点和框”推出效果。

<a id="deepseek-llm-algo-20260726-q8"></a>
##### 8. 题目八：DeepSeek MoE的无辅助损失负载均衡机制推导

**回答：**

这类“无辅助损失”方案的要点不是取消均衡控制，而是把均衡控制从主损失中的可学习惩罚项，改成路由器外部的专家偏置反馈。

设 token $t$ 对专家 $i$ 的原始路由分数为 $s_{t,i}$，专家偏置为 $b_i$。路由选择使用：

$$
\tilde{s}_{t,i}=s_{t,i}+b_i,
\qquad
\mathcal{E}_t=\operatorname{TopK}_i(\tilde{s}_{t,i},k)
$$

关键实现细节是：偏置用于影响 Top-$k$ 的选择，但专家组合权重通常仍使用未加偏置的原始路由分数并做归一化。这样偏置负责纠正“谁被选中”，尽量不直接改变主任务中的混合权重。

在一个统计窗口内，设专家 $i$ 实际接收的 token 数为 $c_i$，平均负载为 $\bar c=\frac{1}{E}\sum_i c_i$。高负载专家应被降低偏置，低负载专家应被提高偏置。一种离散更新可写成：

$$
b_i\leftarrow b_i-\gamma\,\operatorname{sign}(c_i-\bar c)
$$

也可以用带裁剪的比例反馈：

$$
b_i\leftarrow\operatorname{clip}\left(
b_i-\gamma\frac{c_i-\bar c}{\bar c+\epsilon},
b_{min},b_{max}\right)
$$

其中 $\gamma$ 控制反馈速度。偏置一般按 batch 或固定 token 窗口更新，而不是对每个 token 更新，否则噪声会使路由剧烈抖动。实践还要考虑专家容量、跨节点通信、局部负载与全局负载不一致，以及偏置版本在分布式设备之间同步。

与辅助损失相比，外部偏置的优势是不会把一个需要调权重的均衡目标直接加入语言建模损失，降低主任务梯度被干扰的风险；它也能在训练早期通过反馈快速纠正热门专家。代价是它引入了一个非梯度控制回路：$\gamma$、更新窗口、偏置上下界和同步延迟都需要调节；统计窗口太短会抖动，太长又响应不及时；只看 token 数还可能掩盖某些专家计算更慢或通信更远的问题。

辅助损失的典型思想是同时惩罚选择频率和平均路由概率的不均衡，例如让各专家的 `load` 与 `importance` 接近均匀；无辅助损失并不等价于“天然均衡”，它仍然要通过实时负载监控和偏置更新实现闭环控制。最后要评估的不是偏置公式本身，而是负载变异系数、token 丢弃率、All-to-All 时间、训练损失、验证效果和下游质量的综合变化。
