<a id="sota-overview"></a>
# 07 SOTA 模型拆解

## 目录导航

- [1. Qwen：通用 Transformer 到混合注意力与可控推理](<#sota-section-47>)
  - [面试问题：如何用一条正确主线概括 Qwen 的版本演进？](<#sota-section-47-question-01>)
  - [面试问题：Qwen3 与 Qwen3.8 的公开架构分别有什么关键点？](<#sota-section-47-question-02>)
  - [面试问题：DCA、YaRN、RoPE 分别属于哪一代、各自解决什么问题？](<#sota-section-47-question-03>)
  - [面试问题：什么是Dual Chunk Attention?](<#sota-section-47-question-04>)
    
- [2. Kimi：推理时扩展、MoE 与超长上下文](<#sota-section-48>)
  - [面试问题：Kimi K1.5、K2 和 K3 的主线是什么？](<#sota-section-48-question-01>)
  - [面试问题：K2 与 K3 的 MoE、注意力和长上下文怎样比较？](<#sota-section-48-question-02>)
  - [面试问题：什么是Kimi Delta Attention？](<#sota-section-48-question-03>)
    
- [3. DeepSeek：MLA、DeepSeekMoE、推理强化与稀疏注意力](<#sota-section-49>)
  - [面试问题：DeepSeek-V3 的 MLA、MoE、负载均衡和 MTP 如何协同？](<#sota-section-49-question-01>)
  - [面试问题：DeepSeek-R1 为什么要经历 R1-Zero、冷启动和多阶段 RL？](<#sota-section-49-question-02>)
  - [面试问题：什么是 DeepSeek Sparse Attention？](<#sota-section-49-question-03>)
    
- [4. GLM：异步 RL、IndexShare](<#sota-section-50>)
  - [面试问题：GLM-4.5、GLM-5 与 GLM-5.2 的确定性变化是什么？](<#sota-section-50-question-01>)
  - [面试问题：DSA、异步 RL、IndexShare 和 MTP 的工程价值是什么？](<#sota-section-50-question-02>)
  - [面试问题：什么是IndexShare？](<#sota-section-50-question-03>)
    
- [5. GPT、claude：闭源模型Top级模型](<#sota-section-51>)
  - [面试问题：GPT-5.6 的 Sol、Terra、Luna、max 和 ultra 分别是什么？](<#sota-section-51-question-01>)
  - [面试问题：Claude 的混合推理（hybrid reasonin）指什么？](<#sota-section-51-question-02>)
  - [面试问题：Fable 5、Mythos 5 应怎样解释？什么是Claude code？](<#sota-section-51-question-03>)
    
- [6. Grok：从开源 Grok-1 到闭源 Grok 4.6](<#sota-section-52>)
  - [面试问题：Grok 的哪些架构细节可以复现？](<#sota-section-52-question-01>)
  - [面试问题：Grok 4 与 Grok 4.6 的训练主线是什么？](<#sota-section-52-question-02>)
  - [面试问题：Grok Build 和 Grok Bot 与基础模型是什么关系？](<#sota-section-52-question-03>)

---

<a id="sota-section-47"></a>
## 47. Qwen：通用 Transformer 到混合注意力与可控推理

<a id="sota-section-47-question-01"></a>
### 面试问题：如何用一条正确主线概括 Qwen 的版本演进？

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

Qwen 的主线是**持续改善“可训练的通用文本骨干、长上下文、MoE 效率、后训练推理和可部署性”**。

| 版本/分支 | 核心重点 |
| --- | --- |
| Qwen1.5 | 覆盖 0.5B 到 72B，并发布细粒度 MoE 变体 |
| Qwen2 | GQA、RoPE、RMSNorm、SwiGLU 与 YaRN 长度外推，固化了现代 decoder-only 骨干和 GQA 长上下文方案 |
| Qwen2.5 | 更高质量预训练、指令数据和后训练，把数据、指令对齐与 1M 长上下文工程化；Qwen2.5-1M 引入 DCA |
| Qwen3 | 统一 thinking/non-thinking，稠密与 MoE 同时发布，采用四阶段后训练 |
| Qwen3-Next、Qwen3.5、Qwen3.8 | 以混合线性/全注意力探索长上下文与推理效率；Qwen3.8-2.4T-A95B 是开放权重旗舰 |

<a id="sota-section-47-question-02"></a>
### 面试问题：Qwen3 与 Qwen3.8 的公开架构分别有什么关键点？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

Qwen3 的稠密模型沿用 decoder-only Transformer：分组查询注意力（Grouped-Query Attention，GQA）、旋转位置编码（Rotary Position Embedding，RoPE）、预归一化（pre-norm）RMSNorm 和 SwiGLU 前馈层。相对 Qwen2，报告明确了**去除 QKV bias 并引入 QK-Norm**（将 $q,k$ 归一化后再计算注意力，降低极端内积造成的训练不稳定）。

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{\mathrm{Norm}(Q)\mathrm{Norm}(K)^\top}{\sqrt{d_h}}+M\right)V.
$$

Qwen3 的 MoE 版本以**更强的专家专业化与较低激活计算**为目标，代价是专家并行（Expert Parallelism，EP）通信和负载均衡更关键：

- 使用 128 个路由专家、每 token 选择 8 个专家，不使用 Qwen2.5 MoE 的共享专家，并使用全局批次的负载均衡损失。

Qwen3.8-2.4T-A95B 的当前配置为：

- 2.4T 总参数、约 95B 激活参数、92 层、隐藏维 8192、512 个专家且每 token 路由 10 个再加 1 个共享专家；
- 主干按三层 Gated DeltaNet 加一层 Gated Attention 的 3:1 节奏混合，含多 token prediction（MTP）模块。
- 原生上下文是 262,144 token，可扩展到约 1M。

<a id="sota-section-47-question-03"></a>
### 面试问题：DCA、YaRN、RoPE 分别属于哪一代、各自解决什么问题？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

**RoPE 是基础位置编码；YaRN 是 Qwen2 的长度外推方案；DCA（Dual Chunk Attention）是 Qwen2.5-1M 的长上下文推理优化。**

RoPE主要依赖相对距离 $m-n$。但训练长度之外的旋转相位没有被充分覆盖，直接外推会损害注意力模式。YaRN 通过频率插值/外推与 attention scaling 让模型更平稳地适配更长位置；**DCA 则把 token 对分为同块与跨块关系，为不同关系采用不同的位置索引映射，从而缓解超长上下文中的位置分布偏移**。

<a id="sota-section-47-question-04"></a>
### 面试问题：什么是Dual Chunk Attention?

DCA 是一种**位置编码重映射**方案：把长序列切成块，让任意 query–key 对的**相对距离都被压回训练时见过的范围**内，从而在外推时不必重新训练。

- 把序列切块，对每个 query，把 key 分成三类，**分别用不同的位置规则**。
- **三分支各做一次 softmax 再相加**：不是"一个 softmax 配重映射后的位置"，而是三份注意力输出**求和**，属于一种集成式近似。因此它与"用重映射位置的全注意力"并不严格等价。

| 分支 | 覆盖对象 | 位置规则 | 目的 |
| --- | --- | --- | --- |
| **Intra-chunk** | 同块 key | 真实相对距离 $i-j$， $\Delta_{\mathrm{intra}}=i-j$ | 保住**局部高分辨率** |
| **Successive-chunk** | 紧邻前一块 | 真实相对距离 $i-j$ ， $\Delta_{\mathrm{intra}}=i-j$ | 保住**跨块边界处的连续性**，避免边界断裂 |
| **Inter-chunk** | 更早的所有块 | 用**块内位置**重映射， $p_i=(i\bmod l)+l,\qquad p_j=j\bmod l,\qquad \Delta_{\mathrm{inter}}=p_i-p_j$ | 把远距离压缩进训练分布，**防止外推失效** |

<a id="sota-section-48"></a>
## 48. Kimi：推理时扩展、MoE 与超长上下文

<a id="sota-section-48-question-01"></a>
### 面试问题：Kimi K1.5、K2 和 K3 的主线是什么？

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

- Kimi K1.5 的重点是通过强化学习扩展测试时计算：让模型在数学、代码等可验证任务上生成更长、更可检查的推理轨迹。
- K2 把重点推进到大规模 MoE 基础模型和 Agent 工具能力；
- K3 则在开放权重模型中继续增加稀疏规模、混合注意力和 1M 上下文。

| 模型 | 规格/训练重点 | 核心 |
| --- | --- | --- |
| Kimi K1.5 | 聚焦 long-context 与 RL 驱动的推理扩展 | 让模型在数学、代码等可验证任务上生成更长、更可检查的推理轨迹 |
| Kimi K2 | 1T 总参数、32B 激活参数、61 层、384 路由专家、每 token 选 8 个并有 1 个共享专家；MLA、128K 上下文 | 推进到大规模 MoE 基础模型和 Agent 工具能力 |
| Kimi K3 | 2.8T 总参数、104B 激活参数、93 层、896 路由专家、每 token 选 16 个、2 共享专家、1M 上下文 | 增加稀疏规模、混合注意力和 1M 上下文 |

> **注：**K2 的预训练量为 15.5T token。
>
> **注：**K3 默认打开 thinking，并要求多轮工具调用保留完整 assistant 消息中的 `reasoning_content` 和 `tool_calls`。

<a id="sota-section-48-question-02"></a>
### 面试问题：K2 与 K3 的 MoE、注意力和长上下文怎样比较？

**难度评分：⭐⭐⭐⭐⭐ (5/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

#### 1. 架构差异

K2 使用多头潜在注意力（Multi-head Latent Attention，MLA）压缩 KV 表示，降低生成阶段的 Cache 负担；K3 在 69 层 Kimi Delta Attention（KDA）和 24 层 Gated MLA 间混合，并保留一层稠密层。其设计让递归/线性状态承担多数长序列更新，再用较少的全局注意力层恢复高保真内容寻址。

<a id="sota-section-48-question-03"></a>
### 面试问题：什么是Kimi Delta Attention？

#### 1. 定义与状态更新

**Gated DeltaNet 的细粒度门控改进版**：用 **delta 规则（先擦后写）+ 逐通道对角门控** 替换纯叠加式记忆，再与全注意力**按比例混合**使用。

**（1）共同骨架**

$$
S_t=\lambda_t S_{t-1}+k_t v_t^\top,\qquad y_t=q_t^\top S_t
$$

- $S_t\in\mathbb{R}^{d_k\times d_v}$ ：固定大小的状态矩阵，充当"被压缩的历史"
- $k_t,v_t,q_t$ ：当前 token 的 key / value / query
- $\lambda_t$ ：**标量**遗忘（衰减）系数——这是 GLA 的门控形式

**（2）KDA 真实的状态更新**

$$
S_t=\mathrm{Diag}(\alpha_t)\left(I-\beta_t k_t k_t^\top\right)S_{t-1}+\beta_t k_t v_t^\top
$$

- $\mathrm{Diag}(\alpha_t)$ ：**逐通道对角**遗忘门， $d_k\times d_k$ ，作用在 key 指标上
- $\beta_t\in(0,1)$ ：**标量**写入门（步长），控制新关联的写入强度
- $I-\beta_t k_t k_t^\top$ ：沿 $k_t$ 方向的**擦除投影**

#### 2. 为什么 KDA 能提升检索能力？

令状态对当前 key 的预测尽量接近目标 value：

$$
\mathcal{L}_t(S)=\tfrac{1}{2}\left\lVert S^\top k_t-v_t\right\rVert_2^2
$$

对 $S$ 求梯度并在 $S_{t-1}$ 处走一步（步长为 $\beta_t$）：

$$
\nabla_S\mathcal{L}_t=k_t\left(k_t^\top S_{t-1}-v_t^\top\right)
$$

$$
S_t=S_{t-1}-\beta_t\,k_t\left(k_t^\top S_{t-1}-v_t^\top\right)
=\bigl(I-\beta_t k_t k_t^\top\bigr)S_{t-1}+\beta_t k_t v_t^\top
$$

$I-\beta_t k_tk_t^\top$ **只擦除"沿 $k_t$ 方向的旧映射"**，其余方向原样保留，随后再写入新关联 $k_tv_t^\top$：

- $\lVert k_t\rVert=1,\ \beta_t\in(0,1)$ ： $1-\beta_t\in(0,1)$ ，是**收缩式擦除**；
- $\beta_t\lVert k_t\rVert^2>2$ ：特征值 $<-1$，退化为**放大/振荡**，擦除不再稳定。

<a id="sota-section-49"></a>
## 49. DeepSeek：MLA、DeepSeekMoE、推理强化与稀疏注意力

<a id="sota-section-49-question-01"></a>
### 面试问题：DeepSeek-V3 的 MLA、MoE、负载均衡和 MTP 如何协同？

**难度评分：⭐⭐⭐⭐⭐ (5/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

传统 MoE 往往在语言建模损失外加入负载均衡项，防止少数专家过载，却可能与主任务梯度争夺优化方向。V3 的“auxiliary-loss-free”路线改以动态偏置调节路由负载，令专家使用更均匀。

**MLA 压缩 KV 表示以降低 decode 带宽；DeepSeekMoE 用细粒度专家扩大总容量；无辅助损失的负载均衡降低训练目标冲突；MTP 用于更长预测跨度和推测解码。**

<a id="sota-section-49-question-02"></a>
### 面试问题：DeepSeek-R1 为什么要经历 R1-Zero、冷启动和多阶段 RL？

**难度评分：⭐⭐⭐⭐⭐ (5/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

R1-Zero 的实验说明：对可自动验证的任务直接做大规模强化学习，模型可以出现自检、回溯和更长推理链；但纯 RL 轨迹也会出现可读性、语言混杂和格式稳定性问题。R1 因此先加入少量高质量冷启动推理数据，再做面向推理的 RL、拒绝采样与覆盖更广任务的后训练。

组相对策略优化（Group Relative Policy Optimization，GRPO）的常见形式是**在同一题的 $G$ 个采样答案间构造相对优势**，随后用带 KL 约束的策略目标提高 $A_i$ 大的轨迹概率。

$$
A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})+\epsilon}.
$$

<a id="sota-section-49-question-03"></a>

### 面试问题：什么是 DeepSeek Sparse Attention？

**难度评分：⭐⭐⭐⭐⭐ (5/5) | 考察频率：⭐⭐⭐ (3/5)**


DSA 的核心思想是**把"算权重"和"选位置"解耦**：用低维索引器选出该看的 top-k 位置，再用完整维度的注意力在这些位置上精确算权重。

#### 1. 整体结构

![](imgs/47-3.png)

- 闪电索引器：给每个 $s\le t$ 打一个"值不值得看"的分。
- 稀疏主注意力：在被选中的 $k$ 个 token 上做真正的 softmax 注意力。

#### 2. 闪电索引器的数学形式

对每个 query token $t$ 与历史 token $s$：

$$
I_{t,s}=\sum_{j=1}^{H^{I}} w^{I}_{t,j}\cdot\mathrm{ReLU}\left(q^{I}_{t,j}\cdot k^{I}_{s}\right)
$$


- $q^{I}_{t,j}\in\mathbb{R}^{d^{I}}$：第 $j$ 个索引器 head 的 query，由 $h_t$ 低秩线性投影得到 
- $k^{I}_{s}\in\mathbb{R}^{d^{I}}$：**所有索引器 head 共享**的 key（MQA 式，省显存、省带宽）
- $w^{I}_{t,j}\in\mathbb{R}$ ：head 权重（门），由 $h_t$ 投影加标量缩放得到
- $\mathrm{ReLU}(\cdot)$ ：使每项非负，且**单调**——不改变单个 head 内的 top-k 排序 

#### 3. 选择与稀疏注意力

**(1) top-k 选择**

$$
\mathcal{S}_t=\mathrm{Top}-k_{s\le t}\left(\{I_{t,s}\}_{s=1}^{t}\right),\qquad |\mathcal{S}_t|=k
$$

**(2) 稀疏主注意力**

$$
o_t=\sum_{s\in\mathcal{S}_t}\frac{\exp\left(q_t^{\top}k_s/\sqrt{d_h}\right)}{\sum_{s'\in\mathcal{S}_t}\exp\left(q_t^{\top}k_{s'}/\sqrt{d_h}\right)}\,v_s
$$

#### 4. 索引器怎么训练：KL 蒸馏

把**稠密主注意力**的真实分布当成老师，让归一化后的索引器分数去逼近它：

$$
p_{t,s}=\frac{\exp\left(q_t^{\top}k_s/\sqrt{d_h}\right)}{\sum_{s'\in\mathcal{S}_t}\exp\left(q_t^{\top}k_{s'}/\sqrt{d_h}\right)},
\qquad
\mathcal{L}_{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\left(p_{t,:}\,\Big\|\,\mathrm{Softmax}_{s}\bigl(I_{t,:}\bigr)\right)
$$

<a id="sota-section-50"></a>
## 50. GLM：异步 RL、IndexShare

<a id="sota-section-50-question-01"></a>
### 面试问题：GLM-4.5、GLM-5 与 GLM-5.2 的确定性变化是什么？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

- GLM-4.5 是 355B 总参数、32B 激活参数和 23T 预训练 token 的 MoE 模型；
- GLM-5 为 744B 总参数、40B 激活参数和 28.5T token，并引入 DSA 与异步 RL 基础设施。
- GLM-5.2 进一步提供 1M 上下文、IndexShare 稀疏注意力索引复用与改进 MTP。

<a id="sota-section-50-question-02"></a>
### 面试问题：DSA、异步 RL、IndexShare 和 MTP 的工程价值是什么？

**难度评分：⭐⭐⭐⭐⭐ (5/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

- DSA 减少长上下文注意力候选；
- 异步 RL 提高 rollout 到更新的资源利用率；
- IndexShare 避免相邻稀疏层重复构建索引；
- MTP 通过草稿—验证减少每个输出 token 的主模型步数。

<a id="sota-section-50-question-03"></a>
### 面试问题：什么是IndexShare？

IndexShare在DSA的基础上进一步优化索引器。

- **相邻的若干稀疏注意力层共享同一份稀疏索引（top-k 位置集合），避免每层重复构建索引。**
- 每层的 Q/K/V、gather 出来的 KV、以及注意力权重，全部仍各自计算。

**(1) 共享规则**：每 $g$ 层一组，组内只算一次索引

$$
\mathcal{S}^{(l)}_t\equiv\mathcal{S}^{(l_0)}_t,\quad l\in[l_0,\,l_0+g)
$$

**(2) 等价的掩码写法**：

$$
M^{(g)}_{t,s}=\mathbf{1}_{s\in\mathcal{S}^{(l_0)}_t},\qquad
o^{(l)}_t=\mathrm{softmax}\left(\frac{q^{(l)}_t k^{(l)\top}_s}{\sqrt{d_h}}+\log M^{(g)}_{t,s}\right)v^{(l)}_s
$$

<a id="sota-section-51"></a>
## 51. GPT、claude：闭源模型Top级模型

<a id="sota-section-51-question-01"></a>
### 面试问题：GPT-5.6 的 Sol、Terra、Luna、max 和 ultra 分别是什么？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

#### 1. GPT-5.6 的 Sol、Terra、Luna是什么？

GPT-5.6 将 Sol 定位为旗舰、Terra 定位为日常工作中的能力/成本平衡、Luna 定位为最快且成本最低的层级。

#### 2. Codex是什么？

OpenAI 的**代理式编程产品**（CLI / IDE 扩展 / 云端代理）及其背后模型

#### 3. max 和 ultra 是什么？

`max` 是在 ChatGPT Work 和 Codex 中可选择的更高能力设置；`ultra` 是为复杂任务协调多条并行工作流的系统设置。

<a id="sota-section-51-question-02"></a>
### 面试问题：Claude 的混合推理（hybrid reasonin）指什么？

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

hybrid reasoning 指同一产品家族支持**即时回答**与 **extended thinking** 两种工作方式，开发者在速度与推理深度之间取舍。但extended thinking 会增加 token、时延与成本，并使工具调用轨迹变长。

<a id="sota-section-51-question-03"></a>
### 面试问题：Fable 5、Mythos 5 应怎样解释？什么是Claude code？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

#### 1. Fable 5、Mythos 5 应怎样解释？
Fable 5 是可供一般使用的 Mythos-class 模型；对于部分高风险请求，系统可能回退（fallback）到 Claude Opus 4.8，平均触发比例低于 5%；Mythos 5 的访问则具有更严格的项目与安全条件。

#### 2. 什么是Claude code？
Claude Code 是编码 Agent：它在本地或受控环境中读取仓库、编辑文件、运行命令和测试，再把观察结果送回 Claude 模型继续决策。
它的成功取决于基础模型、prompt、工具定义、上下文选择、shell 权限、测试反馈、记忆和终止策略，而非完全由Claude模型决定。

<a id="sota-section-52"></a>
## 52. Grok：从开源 Grok-1 到闭源 Grok 4.6

<a id="sota-section-52-question-01"></a>
### 面试问题：Grok 的哪些架构细节可以复现？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Grok-1 是可复现基线：

- 314B 参数、64 层、8 个专家且每 token 选 2 个、48 个 Q 头/8 个 KV 头、隐藏维 6144、131,072 词表、RoPE 和 8192 上下文；
- 48/8 头配置体现了 GQA：KV 头少于 Q 头以减少 Cache。

<a id="sota-section-52-question-02"></a>
### 面试问题：Grok 4 与 Grok 4.6 的训练主线是什么？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Grok 4 的描述是：在 200,000 GPU 的集群上**扩大推理强化学习，扩大可验证训练数据的领域，并训练原生工具使用和实时搜索**。RL 从数学/代码的可验证奖励拓展到更广任务和工具环境。

Grok 4.6 继续使用补充训练、模型生成的推理/技术数据、高质量工程数据、SFT 轨迹过滤，以及知识工作、编码、内核优化、网页开发等环境中的 Agent RL。

<a id="sota-section-52-question-03"></a>
### 面试问题：Grok Build 和 Grok Bot 与基础模型是什么关系？

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

Grok Build 是面向软件构建的 Agent Harness，Grok Bot 是持续执行任务的 Agent 产品。

它们在模型外提供任务分解、并行 worker、工具会话、文件/浏览器环境、调度和人类审批。相同 Grok 模型在裸 API、Build 和 Bot 中会因上下文管理、工具质量和权限不同而有不同完成率。
