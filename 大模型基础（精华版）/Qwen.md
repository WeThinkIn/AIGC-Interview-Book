
# 目录导航

- [1.Qwen 的版本谱系与命名应该如何理解？](<#1.Qwen 的版本谱系与命名应该如何理解？>)
  - [面试问题：Qwen 系列是否存在研发主线？](<#面试问题：Qwen 系列是否存在研发主线？>)
  - [面试问题：Qwen 各代的核心变化是什么？](<#面试问题：Qwen 各代的核心变化是什么？>)
  - [面试问题：截至 2026 年 8 月，Qwen3.5、Qwen3.6 与 Qwen3.7 应如何区分？](<#面试问题：截至 2026 年 8 月，Qwen3.5、Qwen3.6 与 Qwen3.7 应如何区分？>)
  - [面试问题：Base、Instruct、Thinking、Dense、MoE 和 A3B 分别表示什么？](<#面试问题：Base、Instruct、Thinking、Dense、MoE 和 A3B 分别表示什么？>)
- [2.Qwen 的 Transformer 骨架有哪些关键设计？](<#2.Qwen 的 Transformer 骨架有哪些关键设计？>)
  - [面试问题：Qwen 的共同基础架构是什么？](<#面试问题：Qwen 的共同基础架构是什么？>)
  - [面试问题：Qwen 的 Tokenizer 为什么适合中英与多语言？](<#面试问题：Qwen 的 Tokenizer 为什么适合中英与多语言？>)
  - [面试问题：GQA 如何降低 Qwen 的 KV Cache？](<#面试问题：GQA 如何降低 Qwen 的 KV Cache？>)
  - [面试问题：QK-Norm 与去除 QKV bias 解决什么问题？](<#面试问题：QK-Norm 与去除 QKV bias 解决什么问题？>)
- [3.Qwen 的 MoE 路线如何演化？](<#3.Qwen 的 MoE 路线如何演化？>)
  - [面试问题：Qwen 的 MoE 在数学上如何工作？](<#面试问题：Qwen 的 MoE 在数学上如何工作？>)
  - [面试问题：Qwen1.5、Qwen2、Qwen3 和 Qwen3-Next 的 MoE 有何区别？](<#面试问题：Qwen1.5、Qwen2、Qwen3 和 Qwen3-Next 的 MoE 有何区别？>)
  - [面试问题：总参数、激活参数、FLOPs、显存和延迟是什么关系？](<#面试问题：总参数、激活参数、FLOPs、显存和延迟是什么关系？>)
- [4.Qwen 的预训练数据与训练阶段如何演化？](<#4.Qwen 的预训练数据与训练阶段如何演化？>)
  - [面试问题：从 3T、7T、18T 到 36T，Qwen 提升的只是数据量吗？](<#面试问题：从 3T、7T、18T 到 36T，Qwen 提升的只是数据量吗？>)
  - [面试问题：Qwen3 的三阶段预训练有什么逻辑？](<#面试问题：Qwen3 的三阶段预训练有什么逻辑？>)
  - [面试问题：Qwen3 为什么对小模型使用强到弱蒸馏？](<#面试问题：Qwen3 为什么对小模型使用强到弱蒸馏？>)
- [5.Qwen 如何实现长上下文？](<#5.Qwen 如何实现长上下文？>)
  - [面试问题：RoPE、YaRN 与 DCA 的职责分别是什么？](<#面试问题：RoPE、YaRN 与 DCA 的职责分别是什么？>)
  - [面试问题：Qwen2.5-1M 如何把上下文扩展到 100 万 Token？](<#面试问题：Qwen2.5-1M 如何把上下文扩展到 100 万 Token？>)
  - [面试问题：全注意力在 Prefill、Decode 和 KV Cache 上的复杂度是什么？](<#面试问题：全注意力在 Prefill、Decode 和 KV Cache 上的复杂度是什么？>)
  - [面试问题：为什么标称 1M 不等于有效理解 1M？](<#面试问题：为什么标称 1M 不等于有效理解 1M？>)
- [6.Qwen3-Next 为什么采用 Gated DeltaNet 与全注意力混合架构？](<#6.Qwen3-Next 为什么采用 Gated DeltaNet 与全注意力混合架构？>)
  - [面试问题：Gated DeltaNet 的数学原理是什么？](<#面试问题：Gated DeltaNet 的数学原理是什么？>)
  - [面试问题：为什么采用 3:1 的线性注意力与全注意力混合？](<#面试问题：为什么采用线性注意力与全注意力混合？>)
  - [面试问题：MTP 为什么既能改善训练又能加速推理？](<#面试问题：MTP 为什么既能改善训练又能加速推理？>)
- [7.QwQ 与 Qwen3 的推理后训练如何演化？](<#7.QwQ 与 Qwen3 的推理后训练如何演化？>)
  - [面试问题：QwQ-32B 在 Qwen 推理路线中有什么作用？](<#面试问题：QwQ-32B 在 Qwen 推理路线中有什么作用？>)
  - [面试问题：SFT、DPO、RLHF 与 GRPO 在 Qwen 中分别做什么？](<#面试问题：SFT、DPO、RLHF 与 GRPO 在 Qwen 中分别做什么？>)
  - [面试问题：Qwen3 的四阶段后训练为什么这样安排？](<#面试问题：Qwen3 的四阶段后训练为什么这样安排？>)
  - [面试问题：Thinking、Non-thinking 与 Thinking Budget 到底是什么？](<#面试问题：Thinking、Non-thinking 与 Thinking Budget 到底是什么？>)
- [8.Qwen 的代码、数学、Embedding 与 Reranker 如何选择？](<#8.Qwen 的代码、数学、Embedding 与 Reranker 如何选择？>)
  - [面试问题：Qwen-Coder 为什么强调执行环境？](<#面试问题：Qwen-Coder 为什么强调执行环境？>)
  - [面试问题：Qwen-Math 的 CoT、TIR 与奖励模型如何协作？](<#面试问题：Qwen-Math 的 CoT、TIR 与奖励模型如何协作？>)
  - [面试问题：Qwen3-Embedding 与 Qwen3-Reranker 有什么本质区别？](<#面试问题：Qwen3-Embedding 与 Qwen3-Reranker 有什么本质区别？>)
- [9.如何只从文本侧理解 Qwen-VL、Qwen3.5 与 Qwen3.6？](<#9.如何只从文本侧理解 Qwen-VL、Qwen3.5 与 Qwen3.6？>)
  - [面试问题：视觉信息如何进入 Qwen-VL 并生成文本？](<#面试问题：视觉信息如何进入 Qwen-VL 并生成文本？>)
  - [面试问题：Qwen2.5-VL 与 Qwen3-VL 的文本理解链路有何变化？](<#面试问题：Qwen2.5-VL 与 Qwen3-VL 的文本理解链路有何变化？>)
  - [面试问题：Qwen3.5 为什么既能做纯文本又被称为原生多模态模型？](<#面试问题：Qwen3.5 为什么既能做纯文本又被称为原生多模态模型？>)
- [10.实际项目如何部署、选型和排障？](<#10.实际项目如何部署、选型和排障？>)
  - [面试问题：如何选择合适的 Qwen 模型？](<#面试问题：如何选择合适的 Qwen 模型？>)
  - [面试问题：部署 Qwen 时最容易忽略哪些配置？](<#面试问题：部署 Qwen 时最容易忽略哪些配置？>)
  - [面试问题：效果差、首 Token 慢、生成慢或 OOM 时如何排查？](<#面试问题：效果差、首 Token 慢、生成慢或 OOM 时如何排查？>)
---


<h1 id="1.Qwen 的版本谱系与命名应该如何理解？">1.Qwen 的版本谱系与命名应该如何理解？</h1>

<h2 id="面试问题：Qwen 系列是否存在研发主线？">面试问题：Qwen 系列是否存在研发主线？</h2>

**难度评分：⭐ (1/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

Qwen 是阿里巴巴 Qwen 团队构建的一组基础模型：以自回归语言建模为基础，逐步扩展出通用文本、推理、代码、数学、检索和视觉语言等分支。研发主线：

```text
Qwen
  -> Qwen1.5：模型谱系和部署生态成熟，出现细粒度 MoE
  -> Qwen2：GQA、DCA + YaRN、多语言与 MoE 系统化
  -> Qwen2.5：18T 高质量预训练数据、百万级 SFT、多阶段 RL
  -> QwQ：把可验证数学/代码奖励用于规模化推理 RL
  -> Qwen3：统一 thinking 与 non-thinking，强化 MoE 与多语言
  -> Qwen3-Next：Gated DeltaNet + 全注意力、高稀疏 MoE、MTP
  -> Qwen3.5：在 Qwen3-Next 文本骨干上做原生多模态早融合
  -> Qwen3.6：沿用 Qwen3.5 架构族，重点升级 Agentic Coding 与思考保持
  -> Qwen3.7：最新 API 产品代际；Max 为纯文本接口，Plus 为多模态输入/文本输出
```

因此，Qwen 的演化不是“每代都发明一种新 Transformer”，而是围绕四个矛盾持续迭代：

- **容量与计算成本**：Dense 之外发展 MoE，并提高专家稀疏度。
- **长上下文与推理成本**：从 GQA、RoPE 外推走向稀疏注意力和混合线性注意力。
- **通用回答与深度推理**：从 Chat 对齐、QwQ 推理 RL，发展到 Qwen3 混合思考模式。
- **通用能力与专项可靠性**：通过 Coder、Math、Embedding、VL 以及可执行环境补足任务闭环。

> **注**：Double Chunk Attention (DCA) - 优化长窗口推理；Gated DeltaNet - 将传统自注意力的二次复杂度降为线性，同时通过可学习的遗忘门控来逼近甚至超越全注意力的表达能力

<h2 id="面试问题：Qwen 各代的核心变化是什么？">面试问题：Qwen 各代的核心变化是什么？</h2>

**难度评分：⭐⭐ (2/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

| 版本 | 主要证据 | 核心变化 | 关键词 |
|---|---|---|---|
| Qwen | 技术报告 | 建立中英、代码、数学、ChatML、SFT/RLHF 和工具调用基础 | 完整 LLM/Chat/Agent 起点 |
| Qwen1.5 | 官方博客 | 更完整的尺寸与量化生态，32K，上游进入 Transformers；MoE 引入细粒度专家、upcycling、共享专家 | 工程化过渡代 |
| Qwen2 | 技术报告 | GQA（分组查询注意力）；DCA （双块注意力）+ YaRN（上下文扩展）；0.5B -72B Dense 与 57B - A14B MoE；约 30 种语言 | 推理效率、长上下文、多语言 |
| Qwen2.5 | 技术报告 | 18T 高质量 token，超过 100 万 SFT 样本，多阶段 RL | 数据与后训练系统化放大 |
| QwQ-32B | 官方博客 | 数学答案验证器、代码执行器驱动 outcome-based RL | 推理 RL 桥梁 |
| Qwen3 | 技术报告 | 36T token、119 种语言/方言；QK-Norm；128 专家 top-8；thinking/non-thinking 融合 | 统一快答与深思 |
| Qwen3-Next | 官方模型卡 | 3:1 Gated DeltaNet/全注意力；512 路由专家 top-10 + 1 shared；MTP | 超长上下文与高稀疏 |
| Qwen3.5 | 官方博客/模型卡 | 复用 Qwen3-Next 文本骨干，视觉文本早融合，约 250K 词表，201 种语言/方言；0.8B-397B-A17B 开放权重 | 原生多模态 Agent 基座 |
| Qwen3.6 | 官方仓库/模型卡 | 与 Qwen3.5 共享架构和 `model_type`；开放 27B Dense 与 35B-A3B MoE，强化 Agentic Coding 与 Thinking Preservation | 最新开放权重代际 |
| Qwen3.7 | 官方博客/API 文档 | Max 为纯文本 API，Plus 为多模态输入/文本输出 API；均标称 1M 上下文，未披露可复现架构与开放权重 | 最新产品/API 代际 |

> **注**：截至 2026-08-16，没有检索到题名为 Qwen3.6 或 Qwen3.7 的官方技术报告。

<h2 id="面试问题：截至 2026 年 8 月，Qwen3.5、Qwen3.6 与 Qwen3.7 应如何区分？">面试问题：截至 2026 年 8 月，Qwen3.5、Qwen3.6 与 Qwen3.7 应如何区分？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

##### 1. 先给结论：必须同时回答三个“最新”

截至 2026-08-16：

- **最新公开产品/API 代际**：Qwen3.7。Qwen3.7-Max 于 2026-05-20 发布，当前公开接口为文本输入、文本输出；Qwen3.7-Plus 于 2026-06-01 发布，支持文本/图像/视频输入并输出文本。
- **最新官方开放权重代际**：Qwen3.6。官方开放了 `Qwen3.6-35B-A3B` 与 `Qwen3.6-27B`，并提供相应 FP8 权重。
- **最新可核验的实现架构族**：Qwen3.5/Qwen3.6。Transformers 文档明确说明 Qwen3.6 checkpoint 与 Qwen3.5 共享架构和 `model_type`，使用相同实现类加载。

| 版本 | 发布与供给形态 | 文本侧实现能确认什么 |
|---|---|:-:|
| Qwen3.5 | 0.8B、2B、4B、9B、27B、35B-A3B、122B-A10B、397B-A17B 开放权重 | 原生多模态早融合；文本骨干采用 3:1 Gated DeltaNet/全注意力混合堆叠，Dense 与 MoE 并行 |
| Qwen3.6 | 2026-04；27B Dense、35B-A3B MoE 开放权重 | 沿用 Qwen3.5 模型类型；强化 Agentic Coding，引入可选 Thinking Preservation；保留混合注意力、MTP 和文本模式部署 |
| Qwen3.7-Max | 专有 API/在线服务 | 文本输入、文本输出，官方标称 1M 上下文，面向长程 Agent、代码与办公工作流 |
| Qwen3.7-Plus | 专有 API/在线服务 | 文本/图像/视频输入，文本输出，官方标称 1M 上下文 |

##### 2. Qwen3.6 的可复现实现细节

`Qwen3.6-35B-A3B` 是 MoE：35B 总参数、约 3B 激活参数、40 层，布局为：

```text
10 x [3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE)]
```

每个 MoE 层有 256 个路由专家，每 token 激活 8 个路由专家，再经过 1 个共享专家。全注意力层采用 16 个 Q heads 与 2 个 KV heads；模型训练了多步 MTP**（多 token 预测）**。其稀疏度可近似看成路由专家激活比例 $8/256=3.125\%$（实际每 token 的计算还包括共享专家、注意力、投影、归一化和视觉/文本前处理）。

`Qwen3.6-27B` 是 Dense：27B 参数、64 层，布局为：

```text
16 x [3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN)]
```

它的全注意力层采用 24 个 Q heads 与 4 个 KV heads，Dense FFN 中间维度为 17,408，同样训练了多步 MTP。两款模型的原生上下文均为 262,144 token，模型卡给出的 YaRN 外推上限约为 1,010,000 token。

两者都被定义为带视觉编码器的因果语言模型，但纯文本服务可以走 `--language-model-only` 路径，跳过视觉编码器与多模态 profiling，并把显存留给 KV cache。（**“支持纯文本部署”不等于“checkpoint 本身是纯文本模型”。** ）此外，型号中的 27B/35B 是模型卡的语言模型参数口径；Hugging Face 集合页显示约 28B/36B，可解释为对完整多模态 checkpoint（含视觉侧等参数）的仓库统计口径。

超出 262K 原生窗口时，模型卡使用 YaRN 做位置外推。**倍率调得越大，能支持的长度越长，但会损害短文本的效果。所以生产上不能"一刀切"开最大倍率，而应该按请求长度分流到不同配置的实例**。

##### 3. Qwen系列还有哪些垂直分支？

还有三条值得面试关注的垂直分支：

- **Qwen-AgentWorld**：35B-A3B 与 397B-A17B 语言世界模型，用 CPT -> SFT -> RL 学习环境状态转移，可作为 Agent 训练的环境模拟器或统一 Agent 基座。
- **Qwen-UI-Agent**：统一 GUI 与 CLI 动作空间，使用超过 100 轮轨迹的在线 RL 和大规模并发交互环境，面向移动端、桌面、浏览器与 DeepSearch。
- **Qwen-CUA**：397B-A17B MoE 骨干，仅观察屏幕截图并输出键鼠动作；通过约 4 万个可验证任务和完整轨迹 RL 训练，报告还披露了超过 1T 参数的 Qwen-CUA-Max。


<h2 id="面试问题：Base、Instruct、Thinking、Dense、MoE 和 A3B 分别表示什么？">面试问题：Base、Instruct、Thinking、Dense、MoE 和 A3B 分别表示什么？</h2>

**难度评分：⭐⭐ (2/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

| 名称 | 含义 | 适合场景 | 常见误区 |
|---|---|---|---|
| Base | 完成预训练但未充分对齐的基座模型 | 继续预训练、SFT、领域适配、研究 | 直接拿来聊天可能不会稳定遵循指令 |
| Instruct | 经过指令微调和偏好/强化学习的助手模型 | 对话、抽取、工具调用、生产推理 | 不等于知识更新或绝对可靠 |
| Thinking | 允许生成较长推理轨迹的模式或专门 checkpoint | 数学、代码、复杂规划 | 推理更长不保证结论正确 |
| Non-thinking | 省略或压缩显式思考的快速模式 | 简单问答、摘要、低延迟服务 | 不是 Base 模型 |
| Dense | 每层的主要 FFN 参数对每个 token 都参与计算 | 部署简单、延迟可预测 | 参数增大通常直接增加计算量 |
| MoE | 每个 token 只路由到少数专家 | 用较低激活计算换更大总容量 | 权重存储和通信成本不会消失 |

以 `Qwen3-30B-A3B` 为例：

- `30B` 是总参数量的量级，主要影响权重存储、加载和分布式切分。
- `A3B` 表示每个 token 前向时激活约 3B 参数，较接近单 token 的计算规模。

<h1 id="2.Qwen 的 Transformer 骨架有哪些关键设计？">2.Qwen 的 Transformer 骨架有哪些关键设计？</h1>

<h2 id="面试问题：Qwen 的共同基础架构是什么？">面试问题：Qwen 的共同基础架构是什么？</h2>

**难度评分：⭐⭐ (2/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

##### 1. 自回归目标

Qwen 主线文本模型采用因果 decoder-only 架构。给定 token 序列 $x_{1:T}$，预训练最基本的目标是最小化负对数似然：

$$
\mathcal{L}_{\text{NTP}}(\theta)
=-\sum_{t=1}^{T}\log p_{\theta}(x_t\mid x_{<t}).
$$

因果掩码保证第 $t$ 个位置只能读取 $x_{\le t}$。训练时可以并行计算所有位置的 loss；生成时则必须把新 token 逐步追加到上下文。

##### 2. 典型 Block

Qwen 到 Qwen3 的 Dense 文本骨干大体保持下面的结构：

```text
Token IDs
  -> Token Embedding
  -> N x [RMSNorm -> Causal Attention -> Residual
          RMSNorm -> SwiGLU FFN/MoE -> Residual]
  -> Final Norm
  -> LM Head
  -> Next-token logits
```

关键组件的职责不能混淆：

- **RoPE**：把位置信息作用到 Q/K 的旋转相位中。
- **RMSNorm + Pre-Norm**：控制隐藏状态尺度并改善深层训练稳定性。
- **SwiGLU**：用门控 FFN 增强逐 token 的非线性变换。
- **GQA**：减少 KV head 数，主要优化推理阶段的 KV cache 与带宽。
- **QK-Norm**：控制 Q/K 与 attention logits 的尺度，主要服务训练稳定性。
- **MoE**：替换 Dense FFN，扩大总容量但只激活少量专家。

<h2 id="面试问题：Qwen 的 Tokenizer 为什么适合中英与多语言？">面试问题：Qwen 的 Tokenizer 为什么适合中英与多语言？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Qwen/Qwen2 使用基于字节级 BPE 的 tokenizer。Qwen2 报告给出的规模是 **151,643 个普通 token + 3 个控制 token**；Qwen3 报告给出的词表大小是 **151,669**。不同代际和配置中的普通 token、控制 token、padding 后 embedding 行数口径可能不同。

词表扩大有三组权衡：

1. 序列更短，attention 和 KV cache 成本可能下降。
2. Embedding 与 LM Head 参数随词表增大，内存和计算会增加。
3. 低频 token 可能训练不足，切分粒度也会影响跨语言共享。

Qwen3.5 官方博客披露词表扩大到约 250K，目标之一是提高 201 种语言/方言的编解码效率。这里的收益来自 **序列压缩率、训练覆盖与模型规模的共同平衡**。

<h2 id="面试问题：GQA 如何降低 Qwen 的 KV Cache？">面试问题：GQA 如何降低 Qwen 的 KV Cache？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

标准多头注意力为每个 Query head 配一组 Key/Value head。Qwen2 开始系统采用 GQA，让一组 KV heads 被多组 Query heads 共享：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

若每层有 $n_q$ 个 Query heads、$n_{kv}$ 个 KV heads，序列长度为 $L$、head 维度为 $d_h$、每个元素 $b$ 字节，则单层单样本 KV cache 近似为：

$$
M_{KV}\approx 2L\,n_{kv}\,d_h\,b.
$$

前面的 2 来自 K 和 V。相对同 head 数的 MHA，GQA 的 KV cache 比例约为：

$$
\frac{M_{\text{GQA}}}{M_{\text{MHA}}}\approx\frac{n_{kv}}{n_q}.
$$

例如 Qwen2-72B 报告给出 64 个 Q heads、8 个 KV heads，理论上这一部分 KV cache 约为对应 MHA 的 $1/8$。实际显存还包括 allocator、分页、量化尺度、批处理和框架开销。

> **注**：GQA 降低 KV 存储和读取，不改变精确全注意力 Prefill 的 $O(L^2)$ 主要算术量；它也不是稀疏注意力。

<h2 id="面试问题：QK-Norm 与去除 QKV bias 解决什么问题？">面试问题：QK-Norm 与去除 QKV bias 解决什么问题？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Qwen2 保留 QKV projection bias；Qwen3 去掉 QKV bias，并在注意力中加入 QK-Norm。可用简化式理解：

$$
\hat q=\operatorname{Norm}(q),\qquad
\hat k=\operatorname{Norm}(k),\qquad
s_{ij}=\frac{\hat q_i^\top\hat k_j}{\sqrt{d_h}}.
$$

未经控制时，Q/K 范数增长会把 $s_{ij}$ 推得很大，使 softmax 过度尖锐，梯度和训练稳定性变差。QK-Norm 先控制 Q/K 的尺度，使 logits 的变化更多来自方向和语义匹配，而不是向量范数无界增长。

去掉 bias 本身不是性能提升的充分条件。更准确的说法是：Qwen3 把 **无 QKV bias + QK-Norm** 作为一组注意力稳定性设计，并通过大规模训练验证了该配置。

它与 FlashAttention 不属于同一层面：

- QK-Norm 改变模型计算图和数值行为。
- FlashAttention 在保持精确注意力语义的前提下优化分块、IO 和中间存储。

<h1 id="3.Qwen 的 MoE 路线如何演化？">3.Qwen 的 MoE 路线如何演化？</h1>

<h2 id="面试问题：Qwen 的 MoE 在数学上如何工作？">面试问题：Qwen 的 MoE 在数学上如何工作？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

##### 1. Router 与 Top-K

MoE 通常替换 Transformer block 中的 Dense FFN。对 token 表示 $x$，Router 先产生专家概率：

$$
p=\operatorname{softmax}(W_r x).
$$

只选择概率最高的 $k$ 个路由专家：

$$
y_{\text{routed}}
=\sum_{i\in\operatorname{TopK}(p,k)}\tilde p_iE_i(x),
$$

其中 $\tilde p_i$ 可表示对 Top-K 权重重新归一化后的结果。若存在共享专家，则完整输出还包含：

$$
y=y_{\text{routed}}+\sum_j E^{\text{shared}}_j(x).
$$

共享专家对每个 token 都执行，学习通用模式；路由专家按 token 选择，学习更细的条件化表示。

##### 2. 为什么需要负载均衡

如果 Router 长期把大多数 token 送给少数专家，会发生：

- 热门专家容量溢出、token 被丢弃或排队。
- 冷门专家训练不足，总参数没有得到有效利用。
- 跨设备 all-to-all 通信严重不均衡，尾延迟上升。

因此训练中常加入负载均衡目标，使“被选择的比例”和“路由概率质量”不过度集中。Qwen3 报告披露采用 **global-batch load balancing loss**，目的是在更大统计范围内促进专家专门化和负载均衡。

##### 3. 细粒度专家的意义

在总专家参数和每 token 激活参数相近时，把大 FFN 切为更多小专家，可以提供更多专家组合。例如 top-2/8 只有 $\binom{8}{2}$ 种无序组合，而 top-8/128 的组合空间大得多。组合空间不直接等于模型能力，但为按 token 分工提供了更细的粒度。

<h2 id="面试问题：Qwen1.5、Qwen2、Qwen3 和 Qwen3-Next 的 MoE 有何区别？">面试问题：Qwen1.5、Qwen2、Qwen3 和 Qwen3-Next 的 MoE 有何区别？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

| 版本 | 路由专家 | 每 token 路由激活 | 共享专家 | 核心特点 |
|---|---:|---:|---:|---|
| Qwen1.5-MoE-A2.7B | 60 | 4 | 4 | 共 64 个细粒度专家；由 Qwen-1.8B upcycling，部分随机初始化促进专家分化 |
| Qwen2-57B-A14B | 64 | 8 | 8 | 沿用细粒度与共享/路由专家；由 Qwen2-7B upcycling |
| Qwen3-30B-A3B / 235B-A22B | 128 | 8 | 0 | 去掉 shared experts，引入 global-batch 负载均衡 |
| Qwen3-Next-80B-A3B | 512 | 10 | 1 | 高稀疏 MoE，每层配合混合注意力；数字来自官方模型卡 |
| Qwen3.5-397B-A17B | 512 | 10 | 1 | 延续 Qwen3-Next 路线，专家中间维度增大；数字来自官方模型卡 |

这条演化线并不是“共享专家先被证明无效，后来又恢复”。更合理的理解是：

- 是否使用共享专家，取决于专家粒度、激活预算、数据规模、路由稳定性和系统实现。
- Qwen3 在 128/top-8 配置下选择无共享专家；Qwen3-Next 在 512/top-10 的更高稀疏度下重新加入 1 个共享专家作为通用路径。
- 不同代际的消融条件不同，不能只比较一个组件得出普遍结论。

<h2 id="面试问题：总参数、激活参数、FLOPs、显存和延迟是什么关系？">面试问题：总参数、激活参数、FLOPs、显存和延迟是什么关系？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

这五个量回答不同问题：

| 指标 | 主要含义 | 在 MoE 中的决定因素 |
|---|---|---|
| 总参数 | 模型容量与全部权重规模 | 所有专家 + 注意力 + embedding 等 |
| 激活参数 | 一个 token 实际经过的参数量级 | Top-K 专家 + shared experts + 非 MoE 层 |
| FLOPs | 理论算术量 | 激活路径、序列长度、Prefill/Decode、batch |
| 显存 | 权重、KV cache、activation、workspace | 总权重仍需存放或分片，不能按激活参数计算 |
| 延迟 | 用户实际等待时间 | Kernel、带宽、batch、并行、all-to-all、负载不均衡 |

因此，`80B-A3B` 的正确表达是：**它以约 3B 的激活参数量级执行每个 token，但部署仍要处理约 80B 权重的存储和专家分片。**

MoE 在单卡上未必比同激活规模 Dense 快。若专家分散在多卡，token 需要经历 dispatch、all-to-all、专家计算和 combine；小 batch、路由偏斜或网络较慢时，通信可能主导延迟。

<h1 id="4.Qwen 的预训练数据与训练阶段如何演化？">4.Qwen 的预训练数据与训练阶段如何演化？</h1>

<h2 id="面试问题：从 3T、7T、18T 到 36T，Qwen 提升的只是数据量吗？">面试问题：从 3T、7T、18T 到 36T，Qwen 提升的只是数据量吗？</h2>

**难度评分：⭐⭐ (2/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

不是。公开数字展示了规模，但报告反复强调 **质量、分布、验证与训练阶段**。

Qwen2 报告还给出一个很有价值的反例：团队尝试放宽质量阈值得到 12T 数据，但大模型并未相对 7T 高质量数据显著提升，所以主力规模选择 7T。这说明：

$$
\text{更多 token}\not\Rightarrow\text{更高有效训练信息量}.
$$

Qwen3 的数据流程进一步体现了“模型参与数据工程”：

- 用 Qwen2.5-VL 对 PDF 类文档做文字识别，再用 Qwen2.5 清洗文字。
- 用 Qwen2.5、Qwen2.5-Math、Qwen2.5-Coder 合成教材、问答、指令和代码。
- 对超过 30T token 做教育价值、领域、安全等细粒度标注。
- 通过小型代理模型和消融，在实例级标签上优化混合比例。

<h2 id="面试问题：Qwen3 的三阶段预训练有什么逻辑？">面试问题：Qwen3 的三阶段预训练有什么逻辑？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Qwen3 技术报告披露三阶段预训练：

1. **General Stage**：使用 30T 以上 token，序列长度 4,096，建立语言、世界知识和 119 种语言/方言的基础。
2. **Reasoning Stage**：再训练约 5T 更高质量 token，提高 STEM、代码、推理和合成数据比例，同时加速学习率衰减。
3. **Long-Context Stage**：使用数千亿长上下文 token，把训练长度提高到 32,768；其中 75% 的样本长度在 16,384--32,768，25% 在 4,096--16,384。

逻辑是先广覆盖，再提高单位 token 的推理密度，最后用昂贵的长序列训练做能力迁移。若一开始就全部使用 32K 序列：

- 全注意力算术量会显著增加。
- 有效 batch、数据吞吐和训练稳定性更难控制。
- 大量短文被 padding 或拼接，未必提供足够长距离监督。

<h2 id="面试问题：Qwen3 为什么对小模型使用强到弱蒸馏？">面试问题：Qwen3 为什么对小模型使用强到弱蒸馏？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐ (3/5)**

对小模型直接复制大模型的四阶段 RL 成本高，而且小模型探索空间有限。Qwen3 报告采用强到弱蒸馏，把旗舰模型的知识和行为迁移到小模型，并组合：

- **Off-policy distillation**：学生学习教师预先生成的高质量轨迹。
- **On-policy distillation**：学生先从自己的分布采样，再由教师信号指导，减小训练分布与学生推理分布的偏移。

<h1 id="5.Qwen 如何实现长上下文？">5.Qwen 如何实现长上下文？</h1>

<h2 id="面试问题：RoPE、YaRN 与 DCA 的职责分别是什么？">面试问题：RoPE、YaRN 与 DCA 的职责分别是什么？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

##### 1. RoPE：让 Q/K 的内积感知相对位置

RoPE 对每对隐藏维度做位置相关旋转。用二维子空间表示：

$$
R(m,\omega)=
\begin{bmatrix}
\cos(m\omega)&-\sin(m\omega)\\
\sin(m\omega)&\cos(m\omega)
\end{bmatrix}.
$$

位置 $m,n$ 上的 Q/K 旋转后满足：

$$
(R(m)q)^\top(R(n)k)=q^\top R(n-m)k,
$$

所以 attention score 自然依赖相对距离 $n-m$。但当推理距离远超训练范围时，相位进入分布外区域，模型不一定会使用这些位置。

##### 2. YaRN：重标定 RoPE 的频率与 attention 尺度

直接把所有位置除以同一个倍率会同时破坏短距离分辨率。YaRN 的核心是对不同频率采用分段处理：高频部分更重视局部位置，低频部分承担长距离外推，并结合 attention scaling 稳定注意力熵。

##### 3. DCA：重映射块内和跨块的相对位置

Dual Chunk Attention 把长序列分块：

- 块内 token 使用训练窗口内熟悉的相对位置。
- 跨块 token 使用专门的相对位置映射，使距离仍落在模型较熟悉的范围。

DCA 的重点是 **位置表示和注意力组织**，不是通过少算大量 attention pair 把全注意力变成线性复杂度。Qwen2 报告把 DCA 与 YaRN 配合用于长度外推，两者职责互补。

<h2 id="面试问题：Qwen2.5-1M 如何把上下文扩展到 100 万 Token？">面试问题：Qwen2.5-1M 如何把上下文扩展到 100 万 Token？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

正确答案必须同时覆盖 **能力训练** 和 **推理系统**。

##### 1. 能力侧

- **Long data synthesis**：构造远距离检索、跨段关联和长文本生成样本。
- **Progressive pre-training**：逐步提高序列长度，避免直接跳到 1M 的成本和不稳定性。
- **Multi-stage SFT**：让模型不仅能容纳长输入，还学会遵循长文任务。
- **Length extrapolation**：报告给出可把已有上下文至少扩展四倍、且无需额外训练的外推方法。

##### 2. 计算侧

- **稀疏注意力**：只计算估计为重要的注意力模式，降低 1M Prefill 的主要算术量。
- **Sparsity Refinement**：报告指出原始 MInference 在超过约 400K 时可能出现精度损失，因而利用连续相对位置校准稀疏模式，恢复大部分精度。
- **Chunked Prefill**：把超长 prompt 切块进入引擎，控制峰值 activation 和调度粒度；它本身不自动消除总算术量。
- **Kernel 优化**：BladeLLM 针对稀疏 pattern 优化 GPU kernel。
- **Dynamic Chunked Pipeline Parallelism**：按 chunk 动态安排 pipeline，减少负载不均和气泡。
- **Totally Asynchronous Generator**：让 API、scheduler、model runner 和 decoder 等组件异步衔接，减少串行等待。

<h2 id="面试问题：全注意力在 Prefill、Decode 和 KV Cache 上的复杂度是什么？">面试问题：全注意力在 Prefill、Decode 和 KV Cache 上的复杂度是什么？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

设输入长度为 $L$、隐藏维度为 $d$：

| 阶段/资源 | 精确全注意力的主要量级 | 说明 |
|---|---|---|
| Prefill attention 算术 | $O(L^2d)$ | 所有 query 与此前 key 交互 |
| 朴素 attention score 显存 | $O(L^2)$ | FlashAttention 可避免完整落盘 |
| Decode 每生成一个 token | $O(Ld)$ | 新 query 读取既有 K/V |
| KV cache | $O(Ln_{kv}d_h)$ | 随上下文线性增长，GQA 减小常数 |

FlashAttention 通过 tiling 和 online softmax 把中间矩阵留在更快的片上存储中，显著降低 HBM IO 与 activation 显存，但精确 attention 的 Prefill 算术量仍是二次。

<h2 id="面试问题：为什么标称 1M 不等于有效理解 1M？">面试问题：为什么标称 1M 不等于有效理解 1M？</h2>

**难度评分：⭐⭐ (2/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

“支持 1M”至少可能指三种不同能力：

1. 服务接口能接收该长度而不报错。
2. 模型在 passkey/needle 测试中能从远处召回短答案。
3. 模型能对 1M token 做多证据整合、跨段推理和全局一致生成。

第三种远难于第一、二种。真实长文还会受到：

- Lost in the Middle 与位置偏置。
- 大量相似段落造成的证据混淆。
- 稀疏注意力 pattern 漏掉关键连接。
- 引用正确但推理链错误。
- Prefill 延迟、KV cache、并发下降和调用成本。

<h1 id="6.Qwen3-Next 为什么采用 Gated DeltaNet 与全注意力混合架构？">6.Qwen3-Next 为什么采用 Gated DeltaNet 与全注意力混合架构？</h1>

<h2 id="面试问题：Gated DeltaNet 的数学原理是什么？">面试问题：Gated DeltaNet 的数学原理是什么？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

##### 1. 从全注意力到递归状态

全注意力显式保存所有历史 K/V，并让当前 query 与历史 key 比较。线性注意力的另一种视角是：不保存全部历史，而是把历史压缩到固定形状的状态矩阵 $S_t$，再用当前 query 读取：

$$
o_t=S_tq_t.
$$

最简单的累加状态 $S_t=S_{t-1}+v_tk_t^\top$ 容易不断叠加冲突信息。Delta rule 不直接“再写一遍 value”，而是先计算当前状态对 key 的预测，再写入残差。

##### 2. Gated DeltaNet 的简化更新

把单头状态的更新可写成：

$$
S_t=
\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)
+\beta_t v_tk_t^\top,
$$

其中 $\alpha_t\in(0,1)$ 是遗忘门，$\beta_t\in(0,1)$ 是写入强度。展开后可理解为：

$$
S_t=\alpha_tS_{t-1}
+\beta_t\bigl(v_t-\alpha_tS_{t-1}k_t\bigr)k_t^\top.
$$

这两个部分各有明确含义：

- $\alpha_tS_{t-1}$ 对旧记忆做整体衰减，能快速遗忘不再需要的信息。
- $v_t-\alpha_tS_{t-1}k_t$ 是沿当前 key 方向的预测误差，delta update 只修正相关方向。

若 $k_t$ 做了合适归一化，$I-\beta_tk_tk_t^\top$ 可视为沿 $k_t$ 方向擦除一部分旧映射，再写入 $v_tk_t^\top$。这比无条件累加更能处理覆盖和冲突。

##### 3. 复杂度收益与信息瓶颈

递归推理不需要保留所有历史 K/V，线性注意力层的状态大小主要由 head 维度决定，单步读写不随 $L$ 线性扫描全部历史，因此长序列更高效。

代价是历史被压缩到有限状态，多个相似 key 可能相互覆盖；对“精确找回某个原始 token”这类任务，全注意力的显式内容寻址仍有优势。这正是 Qwen3-Next 不采用纯 Gated DeltaNet 的原因。

<h2 id="面试问题：为什么采用线性注意力与全注意力混合？">面试问题：为什么采用 3:1 的线性注意力与全注意力混合？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Qwen3-Next-80B-A3B 官方模型卡披露 48 层，布局为：

```text
12 x [
  3 x (Gated DeltaNet -> MoE)
  1 x (Gated Attention -> MoE)
]
```

也就是 36 层线性注意力和 12 层全注意力。两类层互补：

| 路径 | 擅长 | 代价/限制 |
|---|---|---|
| Gated DeltaNet | 流式压缩历史、局部与累计状态、长序列高吞吐 | 固定状态有信息瓶颈，精确回忆可能发生干扰 |
| Gated full attention | 对历史 token 做显式内容寻址，恢复全局检索 | Prefill 二次，KV cache 随长度增长 |

3:1 的设计让大多数层避免完整二次 attention，同时周期性用全注意力校正和重新整合全局信息。

面试中最重要的纠错是：**Qwen3-Next 不是纯线性注意力模型。** 因为仍有 12 层全注意力，整个模型的 KV cache 仍随 $L$ 增长，只是常数比 48 层全注意力显著小。其 Prefill 也仍包含这些全注意力层的二次计算。

官方模型卡还披露：

- 总参数约 80B，每 token 激活约 3B。
- 512 个路由专家中激活 10 个，并有 1 个共享专家。
- 原生上下文 262,144，可外推到约 1,010,000。
- 15T token 预训练。
- Instruct checkpoint 只支持 non-thinking，不应因为名字里有 Qwen3 就默认存在 `<think>` 输出。

<h2 id="面试问题：MTP 为什么既能改善训练又能加速推理？">面试问题：MTP 为什么既能改善训练又能加速推理？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐ (3/5)**

Next-token prediction 只让当前位置直接预测 $x_{t+1}$。Multi-Token Prediction（MTP）增加辅助预测头，让共享表示同时预测更远的若干 token。不同实现对未来 token 的条件方式不同，下面是表达训练目标的示意式：

$$
\mathcal L
=\mathcal L_{1}
+\lambda\frac{1}{D}\sum_{j=2}^{D+1}\mathcal L_j,
$$

$$
\mathcal L_j=-\sum_t\log p_{\theta,j}(x_{t+j}\mid h_t,x_{t+1:t+j-1}).
$$

这里 $h_t$ 是主干在位置 $t$ 的隐藏状态；训练时可用教师强制提供中间真实 token。训练收益来自更密的监督：隐藏状态不仅要让下一 token 正确，还要包含对后续局部结构有用的信息。这可以改善数据效率和表示规划能力。

推理时，辅助头可以一次提出多个候选 token，主模型再并行验证，形成 speculative decoding。若一段候选被接受，一次主干前向就能推进多个 token。

必须避免两个误解：

- MTP 不等于主模型在无验证情况下每步直接输出多个 token。
- 加速比取决于候选接受率、验证开销、batch、输出分布和推理框架，不是固定倍数。

<h1 id="7.QwQ 与 Qwen3 的推理后训练如何演化？">7.QwQ 与 Qwen3 的推理后训练如何演化？</h1>

<h2 id="面试问题：QwQ-32B 在 Qwen 推理路线中有什么作用？">面试问题：QwQ-32B 在 Qwen 推理路线中有什么作用？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

QwQ-32B 是 Qwen2.5 与 Qwen3 之间的推理路线桥梁。它证明强基础模型可以通过规模化 RL 显著增强数学、代码和 Agent 推理，而不必只依赖更大的参数量。

官方博客披露两阶段 RL：

1. **数学与代码 RL**：从 cold-start checkpoint 开始。数学用最终答案准确性验证器，代码由执行服务器检查是否通过测试，而不是完全依赖一个学习到的奖励模型。
2. **通用 RL**：再用通用奖励模型和规则验证器改善指令遵循、人类偏好与 Agent 能力，同时尽量保持数学和代码表现。

为什么 outcome-based reward 适合数学和代码？因为奖励更接近客观结果：

$$
r(y)=
\begin{cases}
1,&\text{答案通过验证或代码通过测试},\\
0,&\text{否则}.
\end{cases}
$$

但最终答案正确不代表推理过程可靠，模型可能猜中、利用验证器漏洞或产生不可泛化的过程。解决方法包括更强测试、过程检查、格式约束、难度过滤和保留多样化训练分布。

<h2 id="面试问题：SFT、DPO、RLHF 与 GRPO 在 Qwen 中分别做什么？">面试问题：SFT、DPO、RLHF 与 GRPO 在 Qwen 中分别做什么？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

##### 1. SFT：先建立可用行为

对指令 $x$ 和目标回答 $y$，SFT 最小化回答 token 的负对数似然：

$$
\mathcal L_{\text{SFT}}
=-\sum_{t\in\text{assistant mask}}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

它教会模型 Chat Template、任务格式、工具调用和高质量回答范式。Qwen2 报告披露超过 50 万条 SFT 数据；Qwen2.5 扩展到 100 万条以上。

##### 2. DPO：直接利用偏好对

给定偏好回答 $y_w$ 与拒绝回答 $y_l$，DPO 的核心目标可写为：

$$
\mathcal L_{\text{DPO}}
=-\mathbb E\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right]
\right).
$$

它提高相对参考模型对 preferred response 的相对概率，无需显式训练 critic 和运行完整在线 RL。Qwen2 使用离线 DPO，再以奖励模型做在线阶段。

##### 3. RLHF：奖励模型与策略优化的总流程

RLHF 通常包括偏好数据、奖励模型和策略更新。优势是可利用在线采样修正模型自己的输出分布；代价是奖励模型偏差、训练不稳定、采样昂贵和 reward hacking。

##### 4. GRPO：用组内相对奖励替代单独 critic

对同一问题采样 $G$ 个回答，得到奖励 $r_1,\dots,r_G$，组内标准化优势可简化为：

$$
A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}
{\operatorname{std}(r_{1:G})+\epsilon}.
$$

再用 PPO 风格的重要性比率与 clip 目标更新策略，并加入对参考策略的 KL 约束。GRPO 的关键不是“奖励归一化”四个字，而是同一问题的候选互为基线，从而省去单独训练 value critic，并把优化压力放在相对更好的推理轨迹上。

Qwen3 Reasoning RL 报告披露只选取 3,995 个高质量 query-verifier pairs，但使用大 batch、多 rollout、一定的 off-policy 训练和熵控制提高样本效率。**数据条数少不等于生成样本少**，因为每个 query 会产生大量在线 rollout。

<h2 id="面试问题：Qwen3 的四阶段后训练为什么这样安排？">面试问题：Qwen3 的四阶段后训练为什么这样安排？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

Qwen3 的四阶段按“先形成推理格式，再强化正确性，再融合快答，最后做通用对齐”排列：

##### 阶段 1：Long-CoT Cold Start

- 数据覆盖数学、代码、逻辑和 STEM，并配有可验证答案或测试。
- 先过滤太简单、不可验证或包含多个模糊子问题的 query。
- 使用 QwQ-32B 生成多条候选，再过滤错误、重复、猜测、思考与总结矛盾、语言混乱等轨迹。
- 只用精炼子集和较少训练步，目标是建立推理模式，而不是过早限制探索。

##### 阶段 2：Reasoning RL

- 使用未出现在 cold-start、可学习但尽量困难、领域覆盖广的 query-verifier 对。
- 通过 GRPO、多个 rollout 和可验证奖励提高数学与代码正确率。
- 控制熵，使策略既利用已学模式，又保留探索能力。

##### 阶段 3：Thinking Mode Fusion

- 混合 thinking 与 non-thinking SFT 数据。
- Thinking 数据由阶段 2 模型对阶段 1 query 做 rejection sampling，降低能力回退。
- Non-thinking 数据覆盖指令遵循、写作、问答、角色扮演、多语言、数学和代码。
- 借助 chat template 让同一模型识别 `/think` 与 `/no_think`。

##### 阶段 4：General RL

- 在更广任务上优化指令遵循、格式、偏好、Agent 与通用行为。
- 目标是在不明显破坏推理能力的条件下，让模型成为可用助手，而不只是竞赛解题器。

顺序不能随意交换。若先做通用短回答对齐，再做强推理 RL，模型可能丢失简洁回答风格；若没有 cold start 直接 RL，早期采样质量和格式不稳定，可验证奖励也难以引导完整推理。

<h2 id="面试问题：Thinking、Non-thinking 与 Thinking Budget 到底是什么？">面试问题：Thinking、Non-thinking 与 Thinking Budget 到底是什么？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

- **Thinking mode**：模型先生成 `<think>...</think>` 区域，再生成最终答案，适合多步推理。
- **Non-thinking mode**：chat template 预置空 thinking block 或使用相应控制，让模型直接回答。
- **Thinking budget**：当 thinking token 达到阈值时，外部控制器停止继续思考，插入停止思考提示和 `</think>`，再让模型依据当前状态输出答案。

Qwen3 报告给出的控制路径包括：

- 用户消息或系统消息中的 `/think`、`/no_think`。
- Hugging Face chat template 的 `enable_thinking=False`。
- 多轮对话中以最后出现的模式标记为准。

Thinking budget 不能简单等同于 `max_new_tokens`：

- `max_new_tokens` 是整段输出的硬上限，可能连最终答案一起截断。
- Thinking budget 只约束思考阶段；达到阈值后还要结束思考并生成 final answer。

更多 thinking token 通常提供更大的搜索预算，但收益会饱和，也可能出现反复、自洽地犯错或验证器投机。生产系统应按任务难度路由，并分别评估准确率、平均 reasoning tokens、TTFT、TPOT 与总成本。

<h1 id="8.Qwen 的代码、数学、Embedding 与 Reranker 如何选择？">8.Qwen 的代码、数学、Embedding 与 Reranker 如何选择？</h1>

<h2 id="面试问题：Qwen-Coder 为什么强调执行环境？">面试问题：Qwen-Coder 为什么强调执行环境？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

代码模型的核心矛盾是：**语言相似度不能可靠判断程序是否正确。** 可编译、测试通过、文件修改正确和终端任务完成，才是更强的监督。

Qwen2.5-Coder 基于 Qwen2.5 架构继续预训练超过 5.5T code-related tokens，数据覆盖源码、代码相关文本、合成数据、数学与通用文本，任务包括生成、补全、推理和修复。保留一定数学与通用文本比例，是为了避免专项继续预训练导致能力窄化。

Qwen3-Coder 进一步把代码能力推进到 Agent：模型不只补全函数，还要读仓库、调用工具、编辑多个文件、运行测试并根据错误继续迭代。官方博客披露旗舰模型为 480B-A35B，原生 256K，上下文可外推到 1M，7.5T 训练 token 中约 70% 是代码，并构建 20,000 个可交互环境用于长时程 Agent RL。

Qwen3-Coder-Next 技术报告披露 80B 总参数、约 3B 激活，通过大规模可验证 coding tasks 和 executable environments 进行 mid-training 与 RL。它的关键不是“更会背代码”，而是学习闭环：

```text
观察仓库/终端状态
  -> 规划并调用工具
  -> 修改文件
  -> 编译/运行测试
  -> 根据环境反馈修正
  -> 直到任务完成或预算耗尽
```

报告中的 SWE-Bench、Terminal-Bench 等分数依赖 scaffold、工具集合、最大步数、测试环境和采样参数。比较模型时必须固定 Agent harness，不能只比较模型名。

<h2 id="面试问题：Qwen-Math 的 CoT、TIR 与奖励模型如何协作？">面试问题：Qwen-Math 的 CoT、TIR 与奖励模型如何协作？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Qwen2.5-Math 的主线是贯穿预训练、后训练和推理的 self-improvement：

1. 用前代 Qwen2-Math-Instruct 生成和筛选大规模数学数据。
2. 对同一题大量采样，用正确性与质量信号训练数学奖励模型。
3. 奖励模型筛选下一轮 SFT 数据；更强 SFT 模型再生成更强数据，迭代更新 RM。
4. 在最终 SFT 模型上用 RM 做强化学习。
5. 推理时可用 RM 对多个候选做 rerank 或引导搜索。

**CoT** 让模型显式展开多步推理；**TIR（Tool-Integrated Reasoning）** 允许在推理中调用 Python、计算器等工具处理精确运算。

可以把 TIR 看成策略与环境交互：

$$
\text{state}_t
\xrightarrow{\pi_\theta}\text{tool call}_t
\xrightarrow{\text{executor}}\text{observation}_{t+1}.
$$

它主要降低算术和符号执行错误，但工具不会自动选择正确公式，也不会保证前提正确。

奖励模型也有两类常见粒度：

- **Outcome Reward Model（ORM）**：主要判断最终结果或完整回答。
- **Process Reward Model（PRM）**：对中间步骤给分，更有利于定位错误，但过程标注成本更高，也可能把某一种书写风格当成正确过程。

面试中应主动指出：最终答案验证、过程奖励和工具执行是互补信号，任何一个单独使用都可能被策略钻漏洞。

<h2 id="面试问题：Qwen3-Embedding 与 Qwen3-Reranker 有什么本质区别？">面试问题：Qwen3-Embedding 与 Qwen3-Reranker 有什么本质区别？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

Qwen3 Embedding 系列为 Embedding 和 Reranker 都提供 0.6B、4B、8B 三种规模，支持 100 多种语言、代码与跨语言检索。两者都源自 Qwen3 Dense 基座，但计算范式完全不同。

##### 1. Embedding：独立编码，适合大规模召回

Query 和 document 分别编码：

$$
e_q=f_\theta(q),\qquad e_d=f_\theta(d),
$$

$$
s(q,d)=\frac{e_q^\top e_d}{\lVert e_q\rVert\lVert e_d\rVert}
\quad\text{或}\quad e_q^\top e_d.
$$

文档向量可以离线计算并建立 ANN 索引，在线只编码 query，再检索百万甚至更大规模语料。Qwen3-Embedding 支持 MRL，可在部署时截取不同向量维度，在效果、存储和检索速度之间折中。

##### 2. Reranker：联合编码，适合精排

Reranker 把 instruction、query 和 document 拼成同一序列，让 self-attention 显式建模词级交互。官方模型卡的实现让模型判断文档是否满足 query，并比较最后位置 `yes` 与 `no` token 的 logits：

$$
P(\text{relevant}\mid q,d)
=\frac{e^{z_{yes}}}{e^{z_{yes}}+e^{z_{no}}}.
$$

它无法像 embedding 一样预计算一个与 query 无关的文档向量，每个 query-document pair 都要重新前向，所以成本高但细粒度交互更强。

##### 3. 标准 RAG 组合

```text
Embedding ANN 召回 Top-100
  -> Reranker 联合打分取 Top-5/Top-10
  -> 将证据交给生成模型
```

Embedding 负责高召回与低成本，Reranker 负责提高前列精度。不能用 Reranker 暴力扫描全库，也不能因为 Embedding 分数高就省略业务相关性评估。

<h1 id="9.如何只从文本侧理解 Qwen-VL、Qwen3.5 与 Qwen3.6？">9.如何只从文本侧理解 Qwen-VL、Qwen3.5 与 Qwen3.6？</h1>

<h2 id="面试问题：视觉信息如何进入 Qwen-VL 并生成文本？">面试问题：视觉信息如何进入 Qwen-VL 并生成文本？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

视觉语言模型的文本输出仍然是自回归 next-token prediction。区别在于上下文除文本 token 外，还包含由视觉编码器产生的视觉 token：

```text
图像/视频
  -> Vision Transformer
  -> patch features
  -> merger/projector
  -> visual tokens

文本 -> tokenizer -> text tokens

[visual tokens, text tokens]
  -> Qwen language backbone
  -> 自回归文本答案/坐标/JSON/工具调用
```

训练时，语言模型学习在视觉 token 条件下生成目标文本。它可以输出 OCR 结果、表格结构、bounding box、文档问答或操作指令，但这条链路的输出空间仍是文本 token。

因此，“Qwen-VL 能理解图片并输出文本”与“Qwen-Image 根据文本生成图片”是两类模型。本章只涉及前者。

<h2 id="面试问题：Qwen2.5-VL 与 Qwen3-VL 的文本理解链路有何变化？">面试问题：Qwen2.5-VL 与 Qwen3-VL 的文本理解链路有何变化？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐ (3/5)**

##### Qwen2.5-VL

Qwen2.5-VL 报告强调：

- 从头训练原生动态分辨率 ViT，使不同尺寸图像按实际 patch 数进入模型。
- 视觉编码器使用 Window Attention 降低高分辨率图像的计算成本。
- 对视频加入绝对时间编码，使模型能把事件与秒级时间对应。
- 强化 OCR、表单、发票、表格、图表、版面和坐标/点定位。

对文本任务最重要的意义是：模型不只是“描述画面”，还把视觉内容转成可验证的结构化文本。

##### Qwen3-VL

Qwen3-VL 报告进一步强调：

- **Interleaved MRoPE**：把文本位置与视觉的二维空间、视频时间维度统一到多维旋转位置表示中。
- **DeepStack**：不只把 ViT 最后一层特征交给语言模型，而是把多层视觉特征注入语言模型的不同深度，保留从局部纹理到高层语义的信息。
- **Text-based time alignment**：把视频时间对齐到文本可表达的时间标记，便于事件定位和语言推理。
- 原生 256K 文本/交错多模态上下文，强调多页文档和长视频中的跨片段推理。

视觉 benchmark 提升不自动意味着纯文本 benchmark 同比例提升。选型时仍应单独测试 OCR、版面保持、表格结构、坐标精度和纯文本回答。

<h2 id="面试问题：Qwen3.5 为什么既能做纯文本又被称为原生多模态模型？">面试问题：Qwen3.5 为什么既能做纯文本又被称为原生多模态模型？</h2>

**难度评分：⭐⭐⭐⭐ (4/5) | 考察频率：⭐⭐⭐⭐ (4/5)**

Qwen3.5 不是“文本模型外挂一个视觉适配器”这么简单。官方博客称其从预训练阶段就在交错文本、图像和视频 token 上进行早融合，所以称为原生多模态基础模型。

但它仍有清晰的文本路径：

- 文本骨干复用 Qwen3-Next 的 3:1 Gated DeltaNet/全注意力混合 decoder。
- 视觉塔复用 Qwen3-VL 编码器；纯文本请求不需要视觉输入。
- Transformers 文档支持 text-generation；部分部署可以使用 language-model-only 路径跳过视觉编码器。

以 Qwen3.5-397B-A17B 官方模型卡为例：

- 397B 总参数、17B 激活。
- 60 层，即 15 组 3 个 Gated DeltaNet + 1 个全注意力层。
- 512 个路由专家，top-10，外加 1 个共享专家。
- 原生 262,144 上下文，可外推到约 1,010,000。
- 约 248K padded token embedding，并训练多步 MTP。

这组数字来自官方模型卡，不应归到 Qwen3 技术报告。对于纯文本部署，仍要确认具体 checkpoint、推理框架是否允许不加载或跳过视觉塔；“模型能接收纯文本”不等于视觉参数自动不占显存。

Qwen3.6 延续了同一设计边界。官方 Transformers 文档明确把 `Qwen3.6-27B` 归入 Qwen3.5 的 Dense 实现类，`Qwen3.6-35B-A3B` 模型卡也标记为 `qwen3_5_moe`。因此 Qwen3.6 在文本侧应理解为 **同一原生多模态架构族上的新 checkpoint 与后训练升级**；使用 vLLM 时可用 `--language-model-only` 跳过视觉编码器，但不能由此把 checkpoint 改称为纯文本基础模型。

<h1 id="10.实际项目如何部署、选型和排障？">10.实际项目如何部署、选型和排障？</h1>

<h2 id="面试问题：如何选择合适的 Qwen 模型？">面试问题：如何选择合适的 Qwen 模型？</h2>

**难度评分：⭐⭐ (2/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

不要先问“哪个榜单最高”，而要按约束逐层筛选：

| 需求 | 优先方向 | 关键验证 |
|---|---|---|
| 通用问答、抽取、摘要 | 开放权重优先评估 Qwen3.6/Qwen3.5/Qwen3；托管 API 可评估 Qwen3.7 | 指令遵循、幻觉、JSON 稳定性、P99、数据合规 |
| 数学与复杂推理 | Qwen3 Thinking 或 QwQ/Qwen-Math | 准确率与 reasoning token 成本、验证器覆盖 |
| 代码补全 | Qwen2.5-Coder 等低延迟代码模型 | 语言/框架覆盖、completion latency |
| 仓库修改、终端 Agent | Qwen3-Coder/Coder-Next，或 Qwen3.6/Qwen3.7 API | 固定 scaffold 下的任务成功率、工具错误恢复、总 token 成本 |
| 百万级长文 | Qwen2.5-1M、Qwen3-Next/Qwen3.5/Qwen3.6；托管 Qwen3.7 API | 有效上下文、TTFT、并发、KV 显存或 API 限额 |
| RAG 召回 | Qwen3-Embedding | Recall@K、向量维度、索引成本 |
| RAG 精排 | Qwen3-Reranker | NDCG/MRR、pair 吞吐、候选数量 |
| OCR/文档/图表转文本 | Qwen2.5-VL/Qwen3-VL/Qwen3.5/Qwen3.6，或 Qwen3.7-Plus API | OCR、版面、表格、坐标、长文档 |
| 端侧/单卡 | 小型 Dense 或量化 checkpoint | 真实显存、首 Token、每 token 延迟、量化回退 |

完整选型至少要固定：模型 checkpoint、精度、推理框架、硬件、最大上下文、并发、采样参数、chat template、工具协议和评测集。否则模型对比没有可复现性。

<h2 id="面试问题：部署 Qwen 时最容易忽略哪些配置？">面试问题：部署 Qwen 时最容易忽略哪些配置？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

##### 1. Chat Template

Base 和 Instruct 的输入格式不同。即使都是 Instruct，不同代际对 system/user/assistant、tool call 和 thinking block 的特殊 token 约定也不同。手写字符串容易造成：

- 特殊 token 缺失或重复。
- Assistant generation prompt 错位。
- 多轮角色边界污染。
- `/think` 或工具调用不生效。

应优先使用 checkpoint 自带 tokenizer 的 `apply_chat_template`，并将 template 版本纳入线上配置和回归测试。

##### 2. Thinking 与解析器

服务端要确认：

- checkpoint 是否支持 thinking、non-thinking 或只支持其中一种。
- 推理框架是否配置 reasoning parser。
- API 是把 reasoning 与 final 分字段返回，还是把 `<think>` 混在 content。
- budget 到达后是否还能留出 final answer token。

##### 3. 上下文配置

不要只改 `max_model_len`。还要检查：

- 模型原生窗口和 RoPE scaling 配置。
- 推理框架是否支持对应 attention 实现。
- 最大长度下 KV cache 是否足够。
- 输入长度与最大生成长度之和是否越界。
- 长上下文外推是否需要 YaRN 等参数。

##### 4. MoE 与混合注意力 Kernel

MoE 需要 expert parallel/all-to-all 支持；Qwen3-Next/Qwen3.5/Qwen3.6 的 DeltaNet 快速路径还依赖对应的 fused kernel。框架即使“能加载”，也可能退化到较慢、较耗显存的参考实现。Qwen3.6 模型卡建议使用较新的 SGLang/vLLM，并为 reasoning、tool call 与 MTP 分别配置解析器或 speculative decoding 参数；能成功启动不代表这些能力已经生效。

##### 5. 量化

AWQ/GPTQ/FP8 等 checkpoint 的 kernel 支持、group size、activation dtype、KV cache dtype 可能不同。量化前后要分别评测：

- 困惑度或生成质量。
- 数学/代码/工具参数的精确性。
- TTFT、TPOT 与吞吐。
- 实际显存，而不是只按位宽理论计算。

<h2 id="面试问题：效果差、首 Token 慢、生成慢或 OOM 时如何排查？">面试问题：效果差、首 Token 慢、生成慢或 OOM 时如何排查？</h2>

**难度评分：⭐⭐⭐ (3/5) | 考察频率：⭐⭐⭐⭐⭐ (5/5)**

| 症状 | 第一层定位 | Qwen 相关检查 |
|---|---|---|
| 输出不遵循指令 | Prompt/template/模型类型 | 是否误用 Base；chat template；thinking flag；system/tool 格式 |
| 长文答错 | 有效上下文与证据 | 是否超过训练分布；RoPE scaling；稀疏 pattern；位置分桶评测 |
| TTFT 高 | Prefill | 输入是否过长；全注意力比例；chunked prefill；prefix cache；batch 调度 |
| TPOT 高 | Decode 与内存带宽 | KV cache、GQA、量化、连续批处理；MoE all-to-all；MTP 是否启用 |
| OOM | 权重/KV/activation/workspace | 总参数而非激活参数；max_model_len；并发；KV dtype；视觉塔是否加载 |
| MoE 吞吐低 | 路由与通信 | expert parallel、热门专家、all-to-all 网络、batch 是否过小 |
| Qwen3-Next/3.5/3.6 很慢 | Kernel 回退 | DeltaNet fused kernel 是否可用；是否回退 PyTorch reference；是否误加载视觉塔 |
| RAG 看似相关但答错 | 检索与生成分层 | Embedding recall、Reranker precision、chunk、引用和生成模型分别评估 |

排障顺序应从可观察指标出发：

```text
先固定请求并记录 token 数
  -> 分解 TTFT 与 TPOT
  -> 查看权重/KV/activation 显存
  -> 检查 template 与模型配置
  -> 缩短上下文/降低并发做对照
  -> 再切换 kernel、量化、并行或模型
```

一次只改一个变量。否则“换了模型、框架、量化和 prompt 后变好了”不能说明根因。
