
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
  - [面试问题：为什么采用 3:1 的线性注意力与全注意力混合？](<#面试问题：为什么采用 3:1 的线性注意力与全注意力混合？>)
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
- [11.综合面试题：如何把 Qwen 讲得完整而不过度承诺？](<#11.综合面试题：如何把 Qwen 讲得完整而不过度承诺？>)
- [12.2026-08-16 证据化模型卡与底层技术卡](<#12.2026-08-16 证据化模型卡与底层技术卡>)
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

<h2 id="面试问题：为什么采用 3:1 的线性注意力与全注意力混合？">面试问题：为什么采用 3:1 的线性注意力与全注意力混合？</h2>

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

<h1 id="11.综合面试题：如何把 Qwen 讲得完整而不过度承诺？">11.综合面试题：如何把 Qwen 讲得完整而不过度承诺？</h1>

### 11.1 30 秒回答模板

> Qwen 是一条以 decoder-only Transformer 为文本主干的基础模型谱系。Qwen2 用 GQA、DCA/YaRN 和 MoE 改善推理效率与长上下文；Qwen2.5 把高质量预训练数据扩到 18T，并用百万级 SFT 和多阶段 RL 提高可用性；QwQ 把可验证数学/代码奖励用于推理 RL；Qwen3 再把 thinking 与 non-thinking 统一。后续 Qwen3-Next 用 3:1 Gated DeltaNet/全注意力、高稀疏 MoE 和 MTP 优化超长上下文，Qwen3.5 在这套文本骨干上做原生多模态早融合。Qwen3.6 沿用 Qwen3.5 架构族，重点升级 Agentic Coding 和思考保持，是当前最新开放权重代际；Qwen3.7 是更新的 API 产品代际，但架构未公开。实际选型还要看开放权重或 API、激活参数、KV cache、有效上下文、工具闭环与推理框架。

### 11.2 两分钟回答应包含的逻辑

1. **家族定位**：Base/Instruct、Dense/MoE、Thinking/Non-thinking、专项分支。
2. **架构主线**：BBPE、RoPE、RMSNorm、SwiGLU、GQA；Qwen3 加 QK-Norm；Next 改为混合线性注意力。
3. **训练主线**：3T -> 7T -> 18T -> 36T，但强调质量、实例级标签、合成验证和阶段式训练。
4. **后训练主线**：SFT/DPO/RLHF -> QwQ outcome RL -> Qwen3 四阶段与模式融合。
5. **成本主线**：MoE 的 A 参数只代表激活量级；总权重、通信、KV cache 和长 Prefill 仍要付费。
6. **边界**：标称 1M 不等于 1M 全局推理；报告 benchmark 不等于业务 SLA；模型卡披露不等于技术报告结论。

### 11.3 高频追问与一句话答案

| 追问 | 一句话答案 |
|---|---|
| Qwen3-235B-A22B 只需存 22B 权重吗？ | 不需要；约 235B 总权重仍要存储或分片，22B 是每 token 激活量级。 |
| GQA 把 attention 复杂度变成线性了吗？ | 没有；它减少 KV heads 和 KV cache 常数，不消除全注意力的二次 Prefill。 |
| DCA 是稀疏注意力吗？ | 不应这样概括；DCA 主要通过分块重映射块内/跨块相对位置，Qwen2.5-1M 的稀疏推理是另一条技术线。 |
| Qwen3 所有型号都是 128K 吗？ | 不是；技术报告表格中 0.6B/1.7B 为 32K，4B 及以上和两款 MoE 为 128K。 |
| Thinking budget 就是 `max_new_tokens` 吗？ | 不是；它在思考达到阈值后结束 thinking，再继续生成 final answer。 |
| Thinking 越长越好吗？ | 通常先提升后饱和，还可能反复或走错，需要按任务评估。 |
| Qwen3-Next 是纯线性注意力吗？ | 不是；每 4 层有 1 层全注意力，48 层中有 12 层全注意力。 |
| Qwen3-Next 的 KV cache 与长度无关吗？ | 不是；DeltaNet 层状态近似固定，但全注意力层的 KV cache 仍随长度线性增长。 |
| 截至 2026 年 8 月，最新 Qwen 是哪个？ | 产品/API 是 Qwen3.7；可本地部署的官方开放权重是 Qwen3.6；回答前必须先区分口径。 |
| Qwen3.6 是全新架构吗？ | 现有证据不支持；官方 Transformers 文档说明它与 Qwen3.5 共享架构和 `model_type`，升级重点在 checkpoint、后训练与 Agent 能力。 |
| Qwen3.7 沿用 Qwen3.6 的 3:1 混合注意力吗？ | 官方未披露 Qwen3.7 计算图或权重配置，不能凭版本号推断。 |
| MTP 会不会降低生成质量？ | 训练时是辅助监督；推理时通常由主模型验证候选，质量取决于验证实现而不是盲目接受。 |
| QwQ 与 Qwen3 是什么关系？ | QwQ 是专项推理模型和 RL 探索，Qwen3 把这类推理能力与快速回答融合进同一模型。 |
| Embedding 和 Reranker 能互换吗？ | 不能；Embedding 独立编码适合 ANN 召回，Reranker 联合编码适合少量候选精排。 |
| Qwen-VL 属于生图模型吗？ | 不是本章所述链路；它把视觉内容编码后生成文本、坐标或结构化结果。 |
| 1M 上下文能替代 RAG 吗？ | 不能直接替代；1M 仍有成本、干扰与有效利用问题，RAG 能控制证据规模、更新和引用。 |
| MoE 一定比 Dense 快吗？ | 不一定；通信、路由、batch、kernel 和硬件会决定真实延迟。 |
| 官方模型卡和技术报告冲突时信谁？ | 先确认是否同一 checkpoint 和发布日期；报告解释训练方法，模型卡通常描述具体发布权重与运行配置。 |

### 11.4 判断题：用来检查是否真的理解

1. **“Qwen2 全系列都训练了 7T token。”** 错。0.5B 和 MoE 的报告口径不同。
2. **“FlashAttention 让精确全注意力从 $O(L^2)$ 变成 $O(L)$。”** 错。它主要优化 IO 和中间显存。
3. **“A3B 表示模型只占 3B 参数的显存。”** 错。它是激活参数量级。
4. **“QK-Norm 与 GQA 都主要为减少 KV cache。”** 错。QK-Norm主要控制 logits 尺度和训练稳定性。
5. **“Qwen3 的 non-thinking 相当于换成另一个 Base 模型。”** 错。它是后训练模型中的行为模式。
6. **“只要 RoPE 能外推，模型就能理解任意长文本。”** 错。还需要长数据、行为训练、计算系统与任务评测。
7. **“线性注意力不会遗忘信息。”** 错。固定状态会压缩历史，存在覆盖与容量限制。
8. **“代码通过一个样例就可以作为可靠 RL 奖励。”** 错。弱测试容易产生 reward hacking。
9. **“Reranker 的文档表示可以离线预计算并复用于任意 query。”** 错。联合编码依赖 query。
10. **“Qwen3.5 能处理纯文本，所以它不是多模态模型。”** 错。模态能力与某次请求是否包含图像是两回事。
11. **“Qwen3.7 是最新版本，所以已经可以下载权重本地部署。”** 错。截至 2026-08-16，Qwen3.7 是 API 产品线，最新官方开放权重为 Qwen3.6。

### 11.5 面试中最稳妥的表述方式

推荐使用：

- “Qwen3 技术报告披露……”
- “Qwen3-Next 官方模型卡给出的 checkpoint 配置是……”
- “在报告的硬件和测试设置下观察到……”
- “理论上减少的是 KV cache 常数，实际收益还取决于……”
- “标称窗口与有效上下文需要分开评估……”

避免使用：

- “官方证明该模型在任何任务都更强。”
- “MoE 只需要激活参数对应的显存。”
- “1M 输入肯定可以完整理解。”
- “推理 token 越多答案一定越正确。”
- “API 版本号相同，所以架构也相同。”

<h1 id="12.2026-08-16 证据化模型卡与底层技术卡">12.2026-08-16 证据化模型卡与底层技术卡</h1>

本章是本次更新的面试速查层。每张卡严格按“设计动机、核心架构、训练方法、创新点、性能评测、局限与权衡、面试问题”七要素组织。数字只在对应官方报告/模型卡的评测设置下成立；没有公开数字时明确写“官方未完全公开”。

## 12.1 版本与定位总表

| 系列 | 公开规模/配置 | 关键上下文 | 主要用途 | 证据 |
|---|---|---|---|---|
| Qwen1.5 | 0.5B--110B；另有 Qwen1.5-MoE-A2.7B | 32K | 通用文本、生态和量化过渡 | [官方博客](https://qwenlm.github.io/blog/qwen1.5/) |
| Qwen2 | 0.5B/1.5B/7B/57B-A14B/72B | 32K--128K | GQA、DCA/YaRN、多语言、代码数学 | [技术报告](https://arxiv.org/html/2407.10671) |
| Qwen2.5 | 0.5B--72B dense；API Turbo/Plus 为 MoE | 32K/128K；Turbo 可到 1M | 数据规模、结构化输出、DPO+GRPO | [技术报告](https://arxiv.org/html/2412.15115) |
| Qwen3 | 0.6B--32B dense；30B-A3B/235B-A22B MoE | 32K/128K | thinking 与 non-thinking 统一 | [技术报告](https://arxiv.org/html/2505.09388) |
| Qwen3-Next | 80B-A3B（首个公开模型卡） | 原生 262,144，可扩展 1,010,000 | Gated DeltaNet/全注意力、高稀疏 MoE、MTP | [HF 模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |

## 12.2 Qwen1.5：工程化过渡与细粒度 MoE

**设计动机。** Qwen1.5 的重点不是推倒重做，而是把 Qwen 的尺寸谱系、聊天对齐、32K 上下文、量化格式和 Transformers/vLLM 生态做成可复用的发布基线。

**核心架构。** 公开版本覆盖 0.5B、1.8B、4B、7B、14B、32B、72B、110B；Qwen1.5-MoE-A2.7B 采用多个更细粒度 FFN 专家、稀疏路由和共享专家思想。官方未完整公开每个尺寸的层数、head dim、路由损失和通信实现，不能从模型名反推。

**训练方法。** 官方公开了 base/chat 两类和 DPO、PPO 对齐，但完整优化器、学习率、并行拓扑和数据配比未完全公开；可确认所有模型支持 32K，并提供 GPTQ、AWQ、GGUF。

**创新点。** 细粒度 MoE 使小激活量模型保持较大容量；Transformers >=4.37 原生支持，减少部署摩擦。

**性能评测。** 官方 Qwen1.5-72B base 报告 MMLU 77.5、C-Eval 84.1、GSM8K 79.5、MATH 34.1、HumanEval 41.5；Qwen1.5-MoE-A2.7B 在小模型比较中 MMLU 62.5、C-Eval 79.2。指标来自同一官方页面，不能与不同 shot/提示的第三方榜单直接比较。[官方表格](https://qwenlm.github.io/blog/qwen1.5/)

**局限与权衡。** 32K 是发布时训练/评测口径；MoE 总权重仍需存储，实际收益受路由通信、batch 和 kernel 影响；Qwen2 已在 GQA、多语言和长上下文上系统升级。

**面试问题。**

1. 为什么说 Qwen1.5 是“工程化过渡代”？回答：尺寸/量化/Transformers 生态成熟，架构创新相对克制。
2. A2.7B 是否只需要 2.7B 显存？回答：不是，A2.7B 是每 token 激活量级，总专家权重仍要驻留或分片。
3. Qwen1.5 的 DPO/PPO 与 Qwen2.5 的 GRPO 有何不同？回答：前者主要做偏好对齐；后者把可验证/可比较的组奖励用于在线策略优化。

## 12.3 Qwen2：GQA 与 DCA+YaRN 的系统化落地

**设计动机。** Qwen2 同时解决推理 KV cache、长上下文、多语言覆盖和代码/数学能力不足。关键纠偏：DCA 在 Qwen2 官方博客和技术报告中已经出现，Qwen2.5 是继续使用并扩展，不是首次引入。

**核心架构。** Qwen2 为 decoder-only Transformer，使用 GQA、SwiGLU、RoPE、QKV bias、RMSNorm+Pre-Norm。5 个公开尺寸中 57B-A14B 是 MoE；所有尺寸使用 GQA。Qwen2-7B/72B-Instruct 在 YaRN/DCA 配置下支持 128K，57B-A14B 为 64K，小模型为 32K。[报告架构章节](https://arxiv.org/html/2407.10671)

**训练方法。** 报告披露清洗后数据从约 3T 扩到 7T，0.5B 使用 12T；SFT 超过 500K；后训练按 offline DPO、reward model、online DPO 组织，并使用拒绝采样、代码执行反馈和回译。AdamW 细节、并行规模和完整数据比例未公开。

**创新点。** GQA 将 KV head 数从 H 降为 G，同时保持 Q head；DCA 以 chunk 重建相对位置，YaRN 进行 RoPE 频率/注意力温度校准；多语言训练加入 27 种语言。

**性能评测。** Qwen2-72B base：MMLU 84.2、HumanEval 64.6、GSM8K 89.5、MATH 51.1、C-Eval 91.0；Qwen2-7B base：MMLU 70.3、HumanEval 51.2、GSM8K 79.9、MATH 44.2、C-Eval 83.2。[官方博客附录](https://qwenlm.github.io/blog/qwen2/)

**局限与权衡。** GQA 只降低 KV cache 常数，不改变全注意力 $O(L^2)$ 的 Prefill；DCA 不是简单稀疏注意力，跨块依然要聚合信息；128K 是特定 checkpoint、模板和评测条件下的有效范围。

**面试问题。**

1. DCA 属于 Qwen2 还是 Qwen2.5？回答：Qwen2 首次在官方技术报告中系统化采用，Qwen2.5 和 Qwen3 继续使用。
2. GQA 如何节省显存？回答：每层 KV cache 近似按 `2 * L * n_kv * d_head * bytes` 增长，n_kv 从 n_q 降到分组数。
3. 为什么还需要 YaRN？回答：DCA 重构注意力相对位置，YaRN 调整 RoPE 频率与 logits 温度，两者解决不同失配，可组合。

## 12.4 Qwen2.5：18T 数据与两阶段 RL

**设计动机。** Qwen2.5 的目标是把“更大数据”转化为更稳定的知识、代码、数学、长文本和结构化输出，而不是只增加参数。

**核心架构。** 开放权重为 dense decoder-only：0.5B/1.5B/3B/7B/14B/32B/72B；使用 GQA、SwiGLU、RoPE、QKV bias、RMSNorm+Pre-Norm。词表为 151,643 BBPE regular tokens，控制 token 从 3 增至 22。报告表中 7B/14B/32B/72B 为 128K，0.5B/1.5B/3B 为 32K；API Qwen2.5-Turbo/Plus 是 MoE。[报告架构表](https://arxiv.org/html/2412.15115)

**训练方法。** 预训练由 7T 扩到 18T，使用 Qwen2-Instruct 质量过滤、Math/Coder 专项数据和合成数据。SFT 超 1M 样本、32K 序列、两 epoch，学习率从 $7\times10^{-6}$ 降至 $7\times10^{-7}$，weight decay 0.1、梯度裁剪 1.0。offline DPO 约 150K 偏好对；online RL 使用 GRPO 和 reward model。报告未公开完整 GPU 数、张量并行和每阶段 wall-clock。

**创新点。** 将结构化数据、JSON、长答案、跨语言和系统提示鲁棒性纳入统一后训练；长上下文采用 4K -> 32K 训练、RoPE base 10K -> 1M，并用 YaRN+DCA 推理扩展。

**性能评测。** 官方博客概览 MMLU 85+、HumanEval 85+、MATH 80+；报告以 72B-Instruct 对比 Llama-3-405B-Instruct，并给出各尺寸详细表。概览值不是单一 benchmark 的完整复现实验，应以报告表格和 shot 设置为准。[官方博客](https://qwenlm.github.io/blog/qwen2.5/) · [技术报告](https://arxiv.org/html/2412.15115)

**局限与权衡。** 18T 是 token 数而非全部“有效信息”；GRPO 依赖奖励质量，可能出现格式奖励投机；静态 YaRN factor 对短文本可能损伤，vLLM 需按长文本场景开启。

**面试问题。**

1. 为什么要 offline DPO 后 online GRPO？回答：先用可验证偏好对稳定方向，再用在线奖励适配 truthfulness/helpfulness 等细粒度行为。
2. GRPO 与 PPO 的关键差异？回答：GRPO 用同一问题的一组采样估计相对优势，省去独立 value model，但对组内样本质量和奖励方差敏感。
3. 18T 是否必然优于 7T？回答：不必然，收益来自过滤、配比、合成验证和阶段调度，token 数本身不是充分条件。

## 12.5 Qwen3：统一 thinking/non-thinking

**设计动机。** 将 QwQ 类深思能力和普通聊天能力放在同一 checkpoint，通过模板或预算调度延迟与质量，而不是在两个模型之间切换。

**核心架构。** dense 0.6B/1.7B/4B/8B/14B/32B；MoE 30B-A3B、235B-A22B。继承 GQA、SwiGLU、RoPE、RMSNorm+Pre-Norm，但去除 QKV bias、加入 QK-Norm；MoE 为 128 experts、每 token top-8，移除 shared experts，并加入 global-batch load-balancing loss。词表 151,669。[技术报告](https://arxiv.org/html/2505.09388)

**训练方法。** 预训练约 36T、119 语言：S1 超 30T/4K，S2 约 5T STEM/code/reasoning，S3 数百 B/32K 长上下文。后训练四阶段为 long-CoT cold start、reasoning RL、thinking-mode fusion、general RL；小模型进一步使用强到弱蒸馏。报告未公开全部奖励权重与集群拓扑。

**创新点。** 通过 `<think>`/`</think>` 边界、`enable_thinking`、`/think`、`/no_think` 和 thinking budget 统一行为模式；在模型容量不变时将推理计算变成可调资源。

**性能评测。** Qwen3-235B-A22B base 报告 MMLU 87.81、GSM8K 94.39、MATH 71.84、EvalPlus 77.60；Qwen3-32B base MMLU 83.61、GSM8K 93.40、MATH 61.62、EvalPlus 72.05。官方还报告 Qwen3-30B-A3B 在约 3B 激活量级上超过 QwQ-32B，具体需按统一评测设置理解。[报告表 3/4](https://arxiv.org/html/2505.09388) · [官方博客](https://qwenlm.github.io/blog/qwen3/)

**局限与权衡。** thinking budget 增大通常先提升后饱和；长 CoT 增加延迟、token 成本和 KV cache；MoE 仍需保存总专家权重，且路由通信可能抵消理论 FLOPs 优势。

**面试问题。**

1. non-thinking 是不是换了一个模型？回答：不是，同一后训练 checkpoint 的模式控制；差别主要在生成策略和训练分布。
2. QK-Norm 为什么有用？回答：对 Q/K 向量做 RMS 归一化，限制点积 logits 的尺度，降低长训练和大 head 维度下的 softmax 极端化。
3. Qwen3 MoE 为什么移除 shared experts？回答：官方设计改为 128 专家 top-8 并配合全局 batch 负载均衡，减少共享路径的固定计算；不能说所有 MoE 都不需要 shared expert。

## 12.6 Qwen3-Next：Gated DeltaNet、全注意力与 MTP

**设计动机。** 超长上下文下，全注意力的 KV cache 和 $O(L^2)$ Prefill 成本成为瓶颈；Qwen3-Next 以状态空间/线性注意力承担大部分层，以少量全注意力保持精确检索。

**核心架构。** 80B 总参数、3B 激活、48 层，布局为 `12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))`。Gated Attention 为 16Q/2KV、head 256、RoPE dim 64；Gated DeltaNet 为 32V/16QK、head 128；512 experts、top-10、1 shared、expert intermediate 512。原生 262,144、可扩展 1,010,000。[官方模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

**训练方法。** 模型卡公开“预训练 15T + post-training”，并声明引入稳定化 LayerNorm、MTP；完整优化器、阶段 token 配比、奖励与并行策略未公开。

**创新点。** 3:1 混合布局在计算效率与精确全局注意之间折中；高稀疏 MoE 降低每 token 激活量；MTP 既提供相邻未来 token 辅助监督，也可在服务端做 speculative verification。

**性能评测。** 模型卡中 Qwen3-Next-80B-A3B-Instruct 的 MMLU-Pro 80.6、GPQA 72.9、AIME25 69.5、LiveCodeBench 56.6、IFEval 87.6、BFCL-v3 70.3；官方宣称 base 在 10% 训练成本下超过 Qwen3-32B，超 32K 吞吐约 10 倍，需在同硬件/框架下复核。

**局限与权衡。** DeltaNet 状态近似固定并不意味着无遗忘；全注意力层 KV 仍随长度增长；MTP 在 Transformers 中通常不可用，需 vLLM/SGLang 的专用实现；模型卡的训练细节少于技术报告。

**面试问题。**

1. Qwen3-Next 是纯线性注意力吗？回答：不是，每四层包含一层 Gated Attention。
2. MTP 如何加速？回答：草稿头并行预测多个后续 token，主模型一次验证，接受的 token 越多，单步有效长度越长。
3. 3B active 是否意味着 3B 显存？回答：不是；80B 权重、专家路由、全注意力 KV 和通信缓冲仍需显存/分片。

## 12.7 Qwen-VL 与 Qwen2-VL：视觉 token 到文本 token

**设计动机。** Qwen-VL 解决纯文本模型无法读取图像文字、布局和空间关系的问题；Qwen2-VL 进一步要求动态分辨率、长视频和设备操作。

**核心架构。** Qwen-VL 采用视觉编码器 + vision-language adapter + Qwen 文本 decoder，并以多任务/交错图文输入训练。Qwen2-VL 的官方报告给出可复述的细节：全部变体复用 675M ViT；去掉绝对位置嵌入，改用 2D-RoPE；相邻 2×2 patch 经 MLP merger 压缩为一个视觉 token。以 224×224、patch size 14 为例，16×16 个 patch 合为 8×8=64 个视觉 token，再加 `<|vision_start|>` / `<|vision_end|>` 两个边界 token，共 66 个输入 token。MRoPE 把 RoPE 通道分给时间、高度、宽度：文本三轴 position id 相同，图像的时间轴固定而 h/w 随空间位置变化，视频时间轴随帧推进。开源 checkpoint 名称为 2B/7B，而报告按包含 675M ViT 的总参数记为 2B/8B；这是命名口径差异，不应误写为两个不同模型。[Qwen2-VL 报告](https://arxiv.org/html/2409.12191)

**训练方法。** Qwen-VL 官方描述三阶段：图文预训练、多任务预训练、交错图文 SFT。Qwen2-VL 先只训练 ViT，再解冻端到端训练，最后冻结 ViT 做指令微调；前两阶段约为 600B 与 800B token，总计约 1.4T（图像、视频和纯文本的混合 token），以文本 token 监督。72B 训练使用 DP/TP/PP、ZeRO-1、sequence parallel、activation checkpointing 和 1F1B pipeline；优化器、全量学习率与各 loss 权重官方未完全公开，不能补写。

**创新点。** Qwen2-VL 的 Naive Dynamic Resolution 让输入按原始长宽保留 token 数；MRoPE 把空间和时间位置统一给 decoder。视频按 2 FPS 采样并用深度为 2 的 3D 卷积将相邻帧合并，在每段最多 16,384 个视觉 token 的约束下取得细节和长度的折中；从而把能力从“看图描述”扩展到坐标定位、OCR、图表、20 分钟以上视频和手机/机器人操作。[Qwen2-VL 报告](https://arxiv.org/html/2409.12191)

**性能评测。** Qwen-VL-Plus/Max 官方报告 DocVQA 91.4/93.1、MMMU 45.2/51.4、MathVista 43.3/50.0。Qwen2-VL-72B 在报告表中取得 MMMU 64.5、DocVQA 96.5、ChartQA 88.3、TextVQA 85.5、MME 2482.7、MMBench-EN 86.5；同表 7B 为 MMMU 54.1、DocVQA 94.5，2B 为 41.1、90.1。应在相同版本、提示和视觉分辨率下比较，不能将 API 或 Qwen-VL-Plus 的数字嫁接给开源 checkpoint。[Qwen-VL](https://qwenlm.github.io/blog/qwen-vl/) · [Qwen2-VL 报告](https://arxiv.org/html/2409.12191)

**局限与权衡。** 动态分辨率带来视觉 token 数波动；视频帧采样决定时间覆盖；坐标输出需做图像尺寸映射和格式校验；视觉幻觉不能仅靠语言模型概率解决。

**面试问题。**

1. 视觉 token 如何进入文本 decoder？回答：encoder 提取 patch 特征，merger 映射到 LLM hidden size，再按特殊 token/交错序列拼接。
2. 为什么不能固定把图像 resize 到 224？回答：文档/OCR细节需要高分辨率，动态分辨率在显存与细节间按输入调整。
3. 视频理解核心难点是什么？回答：帧采样、时间位置编码、跨帧长依赖与 token 预算共同决定效果。

## 12.8 Qwen2.5-VL 与 Qwen3-VL：从视觉 Agent 到长上下文多模态

**设计动机。** Qwen2.5-VL 面向文档、GUI 和视频 Agent；Qwen3-VL 要求在提升视觉推理的同时不损失纯文本能力，并原生支持交错 256K。

**核心架构。** Qwen2.5-VL 提供 3B/7B/72B：ViT 统一为 32 层、hidden 1280、16 heads、patch 14；除第 7/15/23/31 层外用 112×112 window attention，2×2 patch 特征经两层 MLP merger 投到 LLM hidden size。LLM 分别为 36/28/80 层、hidden 2048/3584/8192、KV heads 2/4/8；所有变体预训练 4.1T token。与 Qwen2-VL 相比，MRoPE 的时间轴对齐绝对时间而非帧号，因而能表达不同 FPS 的真实时间间隔。Qwen3-VL 提供 dense 2B/4B/8B/32B 与 MoE 30B-A3B/235B-A22B，使用 SigLIP2、MLP merger、Interleaved MRoPE、DeepStack 和文本时间戳。[Qwen2.5-VL 报告](https://arxiv.org/html/2502.13923) · [Qwen3-VL 报告](https://arxiv.org/html/2511.21631)

**训练方法。** Qwen2.5-VL 明确将预训练扩至 4.1T token：视觉预训练 1.5T/8K（仅 ViT）、多模态预训练 2T/8K（ViT+LLM）、长上下文预训练 0.6T/32K（ViT+LLM）；后训练为 SFT 再 DPO。精确优化器和学习率未全部公开。Qwen3-VL 报告给出 S0 merger-only 67B/8K，S1 全参数约 1T/8K，S2 约 1T/32K，S3 100B/262K，后训练为 long-CoT SFT、强到弱蒸馏和 RL。

**创新点。** Qwen2.5-VL 支持一小时以上视频、动态 FPS、绝对时间、稳定 JSON grounding；Qwen3-VL 以 DeepStack 融合 ViT 多层特征，Interleaved MRoPE 平衡 t/h/w 频谱，时间戳 token 避免超长视频的稀疏绝对时间 id。

**性能评测。** Qwen2.5-VL-72B 在 MMBench-EN 88.6、MMStar 70.8、CountBench 93.6、ScreenSpot 87.1、Charades-STA 50.9 mIoU；其 text-only 评测为 MMLU-Pro 71.2、MATH 83.0、GSM8K 95.3。Qwen3-VL-235B-A22B-Instruct 报告 MMLU-Pro 81.8、MMLU-Redux 92.2、GPQA 74.3；Thinking 在 MMLongBench-Doc 为 56.2（Instruct 57.0），工具增强的 fine-grained 评测为 V* 93.7、HRBench-4K 85.3、HRBench-8K 82.3。它们均是各自报告的评测协议，不能用单一榜单代表全部能力。[Qwen2.5-VL 报告](https://arxiv.org/html/2502.13923) · [Qwen3-VL 报告](https://arxiv.org/html/2511.21631)

**局限与权衡。** 256K 交错上下文的视觉 token 数可能远超文本 token；DeepStack 增强融合但增加多层投影和显存；视觉 RL 的奖励设计比文本更难，坐标和时间戳需做后处理校验。

**面试问题。**

1. DeepStack 为什么有效？回答：浅层 ViT 保留边缘/局部细节，深层保留语义，分别注入早期 LLM 层，减少只用最后一层的瓶颈。
2. Qwen3-VL 为什么用文本时间戳？回答：把连续时间转成可学习的离散文本语义，避免长视频中绝对时间 position id 过大且采样分布苛刻。
3. 如何保证多模态训练不损失文本能力？回答：混合 text-only/VL 数据，采用平方根重加权的 per-token loss，并在报告中单独评测 text-centric 任务。

## 12.9 Qwen2-Audio、Qwen2.5-Omni 与 Qwen3-Omni

### Qwen2-Audio

**设计动机。** 让模型直接理解语音、环境声和音乐，并支持“语音中带指令”的 voice chat，减少外置 ASR 的错误传播。**架构**为 Qwen LM + audio encoder；**训练**为多任务音频语言预训练、SFT、DPO，完整 encoder 型号未公开；**创新**是自然语言 prompt 统一语音转写、翻译、声音/音乐分析；**评测**覆盖 LibriSpeech、Common Voice、Fleurs、Aishell2、CoVoST2、Meld、VocalSound、AIR，官方称超过前代和多数 SOTA；**局限**是 7B 单一规模、音频时长与噪声分布受限制；**面试问题**：为何不必先 ASR（端到端保留韵律/非语音声）；DPO 改善什么（偏好和 factuality）；音频 token 如何对齐（encoder 时间步映射到 LLM 序列）。[官方博客](https://qwenlm.github.io/blog/qwen2-audio/)

### Qwen2.5-Omni

**设计动机。** 统一文本、图像、音频、视频输入，并同时流式输出文本和自然语音。**架构**采用 Thinker-Talker：Thinker 为 Transformer decoder + 音频/图像 encoder，Talker 为双轨 AR decoder，TMRoPE 对齐视频与音频时间；**训练**为端到端多模态训练，细粒度优化器/数据量官方未完全公开；**创新**是 chunk/block streaming 和同一历史上下文下的 speech generation；**评测**在 OmniBench 达到同规模综合 SOTA，单模态接近/超过 Qwen2.5-VL-7B 与 Qwen2-Audio；**局限**为 7B 模型的音频/视频 token、流式调度和语音质量会竞争显存；**面试问题**：Thinker 与 Talker 是否独立（不是，共享历史并端到端训练）；TMRoPE 解决什么（多模态时间同步）；为何输入输出不能简单串联（需要跨模态状态与低延迟调度）。[官方博客](https://qwenlm.github.io/blog/qwen2.5-omni/)

### Qwen3-Omni

**设计动机。** 在文本、图像、音频、视频上保持与单模态模型相当的能力，同时提供实时语音。**架构**为 Thinker-Talker MoE，Talker 采用多 codebook speech codec + causal ConvNet；**训练**包含感知、Talker、Captioner 的分阶段 post-training，具体 token 量和 RL 配方未完全公开；**创新**支持 Thinking、Captioner，40 分钟音频，理论首包延迟 234ms；**评测**36 个音频/音视 benchmark 中 32 个开源 SOTA、22 个总体 SOTA，文本 119 语言、语音理解 19、生成 10；**局限**为多路流式调度复杂、语音 codec 错误会级联、指标跨模态不可简单平均；**面试问题**：为何多 codebook（同时编码语音内容/韵律）；为何用 causal ConvNet（首包低延迟）；如何验证“无模态退化”（与同尺寸单模态模型逐任务对比，而非只看综合分）。[技术报告](https://arxiv.org/html/2509.17765)

## 12.10 CodeQwen、Qwen2.5-Coder 与代码可靠性

**设计动机。** 通用模型会写代码不等于能在真实仓库中定位、修改、运行和修复；代码模型要把语法、仓库上下文、测试执行和 Agent 工具闭环放在一起。

**核心架构。** CodeQwen1.5 是 Qwen1.5 的代码专项分支；Qwen2.5-Coder 为 0.5/1.5/3/7/14/32B，32B 卡片给出 32.5B 参数、64 层、GQA 40Q/8KV、RoPE/SwiGLU/RMSNorm/QKV bias、131,072 full context。[HF 卡片](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)

**训练方法。** Qwen2.5-Coder 使用约 5.5T source code、text-code grounding 与 synthetic 数据，配合预训练和 post-training；多语言代码 sandbox、静态检查和单元测试用于构造高质量指令/奖励。完整代码语言配比、优化器和集群并行未公开。

**创新点。** 从代码补全扩展到 code reasoning、fixing、repo-level context 和 code agent；保持数学与通用能力，不把模型退化成单一语法预测器。

**性能评测。** 官方模型卡称 32B 在代码能力上匹配 GPT-4o；应结合 HumanEval/EvalPlus、MultiPL-E、LiveCodeBench 和真实仓库测试复核，不能只引用“匹配”宣传语。

**局限与权衡。** 通过公开单元测试不保证安全或生产正确；长仓库会增加检索和 KV 成本；工具权限、依赖网络和测试覆盖率决定 Agent 成功率。

**面试问题。**

1. 为什么代码 RL 要执行器？回答：最终正确性是离散可验证的，执行结果比语言 RM 更能抑制语法/逻辑错误。
2. 代码模型为何还要保留通用数据？回答：需求理解、解释、规划和跨语言沟通需要通用语言能力。
3. 如何评估代码 Agent？回答：除 HumanEval 外，使用隐藏测试、仓库级 patch、编译/单测通过率、回归率和安全扫描。

## 12.11 QwQ-32B 与 Qwen3 Thinking：可验证奖励到统一推理

### QwQ-32B

**设计动机。** 探索在强基础模型上扩大 inference-time reasoning RL，先解决可验证数学/代码，再迁移到一般 Agent。

**核心架构。** 32B dense Qwen 系列语言模型；官方未公开独立新 Transformer 结构，重点是 post-training checkpoint。

**训练方法。** cold-start checkpoint；第一阶段 outcome-based RL 使用数学答案验证器和代码执行服务器，第二阶段使用 general reward model 与规则验证器。[官方博客](https://qwenlm.github.io/blog/qwq-32b/)

**创新点。** 把奖励从“语言偏好”推进到最终答案正确性，并加入工具和环境反馈。

**性能评测。** 官方将其与 DeepSeek-R1、DeepSeek-R1-Distill、o1-mini 等比较；具体图表依赖官方 prompt/采样预算，不能把单个分数当作普适结论。

**局限与权衡。** verifier 覆盖范围有限，数学/代码奖励可能导致领域过拟合；长思考增加延迟与成本；工具安全边界仍需宿主系统控制。

**面试问题。**

1. outcome reward 和 process reward 区别？回答：前者只检查最终结果，稀疏但客观；后者逐步评价，密集但标注/模型偏差更大。
2. 为什么先数学代码再 general RL？回答：先用高精度 verifier 建立探索能力，再用一般 RM 修正交互和偏好。
3. QwQ 与 Qwen3 的关系？回答：QwQ 是专项推理 RL 探索，Qwen3 将其能力与 non-thinking 融合。

### Qwen3 Thinking

**设计动机。** 让同一模型按任务动态分配推理 token；**架构**仍是 Qwen3 dense/MoE，区别在模式控制；**训练**long-CoT cold start -> reasoning RL -> fusion -> general RL；**创新**thinking budget 和 `/think`/`/no_think`；**评测**Qwen3 报告在 AIME、LiveCodeBench、BFCL 等显示随 budget 增加而提升；**局限**存在预算饱和、重复思考和 reward hacking；**面试问题**：预算是否等于 max_new_tokens（不是，预算控制 thinking 阶段）；为什么融合非思考数据（降低简单任务延迟）；如何做线上路由（按任务难度、置信度和成本策略分配）。

## 12.12 Qwen-Agent：模型之外的工具闭环

**设计动机。** 模型输出函数名不等于 Agent 完成任务，需要模板、解析器、工具执行、记忆、RAG、代码沙箱和错误重试。

**核心架构。** Qwen-Agent 提供 `BaseChatModel`、`BaseTool`、`Agent`/`Assistant`，支持 Function Calling、MCP、RAG、Code Interpreter、Browser；可连接 DashScope 或 vLLM/OpenAI-compatible 服务。[官方仓库](https://github.com/QwenLM/Qwen-Agent)

**训练方法。** 它是应用框架，不是单一模型训练配方；模型侧使用各 Qwen checkpoint 的 tool-call 后训练，框架侧通过 schema、parser、执行反馈和状态机完成闭环。Agent 训练数据/优化器由具体模型决定。

**创新点。** 统一支持 Qwen2.5、QwQ、Qwen3、Qwen3-VL、Qwen3-Coder 等；支持并行函数调用、MCP 和 Docker code interpreter。

**性能评测。** 官方仓库提供 tool-call demo、DeepPlanning 与各模型 benchmark 入口，但框架没有一个可代表所有模型的单一分数；应报告任务成功率、工具选择准确率、参数合法率、恢复率和成本。

**局限与权衡。** Docker 代码执行仍需权限、网络和文件挂载审计；MCP 工具供应链可能带来提示注入和数据外泄；解析器与 chat template 不匹配会造成“模型会调用但服务失败”。

**面试问题。**

1. Agent 与 function calling 的边界？回答：function calling 是单步协议，Agent 还包括规划、记忆、执行、观察、重试和停止条件。
2. Qwen3 与 Qwen3-Coder 的 parser 是否一样？回答：不一定，仓库建议按模型选择 parser/raw API，不能只复制一套模板。
3. 代码解释器如何做安全隔离？回答：容器/沙箱、只挂载工作目录、资源/网络/系统调用限制和审计；Docker 默认隔离不是完整安全证明。

## 12.13 底层技术面试卡

### RoPE、YaRN、Dynamic NTK 与 DCA

**设计动机。** RoPE 让 $q_m^Tk_n$ 依赖相对位置，但训练窗口外出现未见过的相对距离；长上下文技术要同时保留局部高频和全局低频。

**核心架构/数学。** RoPE 将偶数维拆成复数并乘 $e^{im\theta_d}$，其中 $\theta_d=b^{-2d/D}$；内积自然产生 $m-n$ 的相对位置信息。[YaRN 全文](https://arxiv.org/html/2309.00071) YaRN 组合 NTK-by-parts 频率插值与 attention temperature；Dynamic NTK 在每次 forward 按当前长度更新 scale。DCA 不改 RoPE 频率，而把序列分块，分别计算 intra/inter/successive chunk 的相对位置。[DCA 全文](https://arxiv.org/html/2402.17463)

**训练方法。** YaRN 可用少量长数据微调，Dynamic NTK/DCA 可训练免费；Qwen2/2.5/3 的长上下文仍配合长样本预训练，不能只靠推理 patch。

**创新点。** YaRN 处理频谱/温度，DCA 处理相对位置矩阵；二者正交可组合。

**性能评测。** YaRN 论文报告相对方法约 10x 更少 token、2.5x 更少训练步骤；DCA 论文在 Llama2-70B 展示超过 100K 的训练免费外推。Qwen 的 128K/1M 数字是具体 checkpoint 的官方评测，不等同于论文模型的通用保证。

**局限与权衡。** 长上下文 benchmark 常是 needle/特定任务；动态 RoPE 与 KV cache 顺序不当会错位；DCA 的 chunk size、局部窗口和 FlashAttention 实现决定效果。

**面试问题。**

1. NTK-aware 为什么不等比例缩放？回答：高频维承载局部相对距离，低频维承载长程趋势，应分频率施加压力。
2. DCA 是稀疏注意力吗？回答：它重映射块内/块间相对位置，仍需跨块聚合，不能简单等同 block-sparse。
3. 为什么 1M 不等于有效 1M？回答：位置外推、信息压缩、注意力干扰、检索/任务分布与系统内存都可能先成为瓶颈。

### MHA、MQA、GQA、MLA 与 Qwen 的取舍

**设计动机。** Decode 阶段瓶颈通常是读取 KV cache；减少 KV head 能降低带宽和显存。

**核心架构。** MHA 为每个 Q head 配独立 K/V；MQA 共享单组 K/V；GQA 将 Q heads 分成组共享 K/V。若 Q head 数为 $h_q$、KV head 数为 $h_{kv}$，KV cache 近似从 $2Lh_qd$ 降到 $2Lh_{kv}d$。MLA 通过低秩 latent 压缩 KV，常见于其他模型；Qwen2/Qwen3/Next 公开资料没有把 MLA 作为主线模块，应标为“对比技术，非 Qwen 主线”。

**训练方法。** attention 形式在预训练中端到端学习；Qwen2 起所有尺寸采用 GQA，Qwen3-Next 的 full attention 为 16Q/2KV。具体 head dim 以 checkpoint 为准。

**创新点。** Qwen 用 GQA 与 RoPE、QK-Norm、混合线性注意力组合，兼顾质量和服务成本。

**性能评测。** Qwen2 官方明确 all sizes GQA 并报告更快/更省推理；不要把 GQA 的收益写成精确 attention 复杂度下降。

**局限与权衡。** KV 共享可能损失表达能力；MLA 的压缩/解压 kernel 复杂；GQA 对长上下文仍有全注意力 $O(L^2)$ Prefill。

**面试问题。**

1. GQA 是否改变 Q 的数量？回答：通常保留 Q heads，只减少 K/V heads。
2. 为什么 MQA 不总是最好？回答：极端共享可能损失 head 间多样性，GQA 是质量/成本折中。
3. Qwen3-Next 的 DeltaNet 层还需要 KV cache 吗？回答：主要保存递归状态，但 full-attention 层仍保存随长度增长的 KV。

### SwiGLU、GeGLU、RMSNorm 与 Pre-Norm

**设计动机。** FFN 需要更强非线性，归一化需要稳定深层残差训练。

**核心数学。** SwiGLU 可写为 $\operatorname{SwiGLU}(x)=(xW_a)\odot\operatorname{SiLU}(xW_b)$ 再乘 $W_c$；GeGLU 将 SiLU 换为 GELU。RMSNorm 为 $x/\sqrt{\operatorname{mean}(x^2)+\epsilon}\odot g$，不计算均值。Pre-Norm 在 attention/FFN 前归一化，残差路径梯度更稳；Post-Norm 归一化在残差相加后。

**训练方法。** Qwen2/Qwen2.5/Qwen3/CodeQwen公开使用 SwiGLU、RMSNorm、Pre-Norm；激活维度、epsilon、学习率和 warmup 按 checkpoint/报告，未公开处不能臆造。

**创新点。** Qwen3 在原有组合上加入 QK-Norm，解决 attention logits 稳定性，不是把 RMSNorm 换成别的归一化。

**性能评测。** 这些是架构组件，官方没有单独给出跨模型因果分数；只能引用完整模型 ablation（若报告提供）。

**局限与权衡。** SwiGLU 通常需要两路线性投影，参数/算力高于单门 FFN；RMSNorm 不去均值，极端分布仍需其他稳定化；Pre-Norm 可能影响表示尺度。

**面试问题。**

1. SwiGLU 为什么常优于 ReLU FFN？回答：门控支路按 token 调节信息流，SiLU 平滑且梯度更友好。
2. RMSNorm 与 LayerNorm 差异？回答：RMSNorm 省均值统计和部分算力，但不消除均值偏移。
3. QK-Norm 与 RMSNorm 是一回事吗？回答：不是，前者归一化 attention 的 Q/K 以控 logits，后者归一化残差 hidden state。

### MoE：路由、容量与真实成本

**设计动机。** 用大总容量提升知识与专家专门化，同时只激活少数专家降低每 token FLOPs。

**核心数学。** 对 token 表示 $x$，router 产生 $p=\operatorname{softmax}(W_rx)$，取 top-$k$ 专家 $E_i$，输出 $y=\sum_{i\in TopK(p)}\tilde p_iE_i(x)$；容量因子、丢 token、负载均衡损失决定稳定性。

**训练方法。** Qwen1.5-MoE 使用细粒度专家/共享路径；Qwen2 有 57B-A14B；Qwen3 为 128 experts/top-8/global-batch balance、无 shared；Qwen3-Next 为 512 experts/top-10+1 shared。路由温度、capacity factor、专家并行通信未完全公开。

**创新点。** 专家粒度逐代变细、激活比例降低；Qwen3 报告称 MoE 可用约 1/5 激活参数达到相近 dense 能力。

**性能评测。** 应同时报告 total/active 参数、吞吐、通信和显存；Qwen3-235B-A22B 的 235B/22B 是典型例子。

**局限与权衡。** 总权重存储、all-to-all 通信、负载倾斜、batch 太小时专家利用不足；MoE 不保证单请求低延迟。

**面试问题。**

1. active 参数和 FLOPs 是否相等？回答：通常相关但不相等，还要计 router、共享专家、attention、embedding 和通信。
2. 负载均衡损失解决什么？回答：防止 token 集中到少数专家导致溢出/热点；全局 batch 统计比单卡局部统计更稳。
3. 为什么 MoE 仍可能 OOM？回答：总专家权重、KV cache、all-to-all buffer 和并发请求共同决定峰值显存。

### 训练、对齐与奖励设计

**设计动机。** 预训练学知识，SFT 学格式和任务，偏好/RL 学 helpfulness、可靠性和可验证推理。

**核心目标。** 预训练最小化 $-\sum_t\log p_\theta(x_t|x_{<t})$；SFT 是带 mask 的条件交叉熵；DPO 直接优化 chosen/rejected 的相对 log-ratio；GRPO 对同一 prompt 的多条样本按组相对奖励估计优势；PPO 通过 clipped ratio 与 value model 控制更新。

**训练方法。** Qwen1.5 公开 DPO/PPO；Qwen2 为 SFT+RM+online DPO；Qwen2.5 为百万级 SFT、offline DPO、online GRPO；QwQ 使用数学 verifier/代码执行器 outcome RL；Qwen3 使用 CoT cold start、reasoning RL、融合、general RL 与蒸馏。

**创新点。** 奖励由主观偏好逐步扩展到 execution/verifier/environment feedback，并将 thinking budget 暴露给推理时调度。

**性能评测。** 需要按任务使用 MMLU/C-Eval、GSM8K/MATH、HumanEval/EvalPlus、BFCL、IFEval 与人工偏好；跨报告分数必须核对 shot、模板、解码和 checkpoint。

**局限与权衡。** 奖励模型偏差、verifier 漏测、reward hacking、过度思考和 alignment tax 都可能导致离线分数与业务质量不一致。

**面试问题。**

1. DPO 为什么不需要在线采样？回答：用固定偏好对直接优化隐式奖励，但覆盖和分布滞后可能限制探索。
2. GRPO 为什么适合数学？回答：同题多采样可按最终正确率组内归一化，减少 value model 需求。
3. 如何防止代码执行奖励被 hack？回答：隐藏测试、多样输入、静态安全检查、资源限制和人工抽检联合使用。

### 量化、KV Cache、PagedAttention 与 vLLM

**设计动机。** 权重和 KV cache 是部署的主要内存项；量化降低带宽，分页管理提高并发。

**核心方法。** GPTQ 是近似二阶信息的逐层 weight-only PTQ；AWQ 根据 activation 找 salient channels 并做等价缩放；KV cache 可用 FP8/INT8/INT4（具体 Qwen checkpoint 支持情况看模型卡）；PagedAttention 将 KV 按固定 token block 映射到非连续物理页。[GPTQ](https://arxiv.org/html/2210.17323) · [AWQ](https://arxiv.org/html/2306.00978) · [vLLM](https://arxiv.org/html/2309.06180)

**训练方法。** GPTQ/AWQ 是 post-training calibration，不需要全量再训练；校准集应覆盖中英、代码、长文本和目标领域。vLLM 是 serving 系统，不改变模型权重。

**创新点。** GPTQ 用 Hessian 近似做误差补偿；AWQ 保护约 0.1%-1% activation-salient channels 而保持硬件友好；PagedAttention 通过 block table 支持动态增长、prefix sharing 和迭代级调度。

**性能评测。** GPTQ 论文展示 175B 压到 3/4 bit；AWQ/TinyChat 报告约 3x 以上 FP16 加速；vLLM 论文报告同延迟 2-4x throughput。Qwen 实际速度需用目标 GPU、batch、上下文和量化 kernel 重测。

**局限与权衡。** 低比特会损失数学/代码和长上下文稳定性；KV cache 量化需处理 outlier；分页减少碎片但不减少理论 KV 元素；vLLM/SGLang 对新 Qwen3-Next/MTP 的版本兼容必须锁定。

**面试问题。**

1. GPTQ 与 AWQ 如何选？回答：GPTQ 重构精度强但校准依赖更明显；AWQ activation-aware、泛化和硬件 kernel 友好，最终以目标任务实测。
2. PagedAttention 是否改变 attention 数学结果？回答：只改变 KV 存储与访问布局，理论 attention 结果不变。
3. 为什么长上下文服务先 OOM？回答：KV cache 随层数、KV heads、head dim、序列长度和并发线性增长，权重常常反而不是第一瓶颈。

## 12.14 本次更新的面试结论

1. **关于 DCA 的准确说法**：DCA 在 Qwen2 已经出现；Qwen2.5 和 Qwen3 把它与 YaRN、长上下文预训练继续组合，Qwen3-Next 则进一步用混合线性/全注意力降低超长上下文成本。
2. **关于“最新”**：本文件覆盖截至 2026-08-16 可公开核实的 Qwen3-Next、Qwen3.5/3.6/3.7 产品信息；Qwen3.7 的公开资料主要是 API 产品页，架构与权重未公开，不把版本号当作架构证据。
3. **关于评测**：任何“超过”都应补充 checkpoint、提示模板、shot、解码、硬件、上下文长度和是否使用工具；官方宣传页、技术报告和第三方榜单不能无条件互换。

<h1 id="13.主要参考资料">13.主要参考资料</h1>

### 13.1 主线技术报告与官方材料

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/)
- [Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters](https://qwenlm.github.io/blog/qwen-moe/)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3-Next-80B-A3B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3.5-397B-A17B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [Hugging Face Transformers: Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.6 官方仓库](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6 官方 Hugging Face 集合](https://huggingface.co/collections/Qwen/qwen36)
- [Qwen3.6-35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.7-Max 官方发布入口](https://qwen.ai/home)
- [Qwen API 平台：Qwen3.7-Max / Qwen3.7-Plus](https://qwen.ai/apiplatform)
- [Alibaba Cloud Model Studio: Qwen3.7-Max](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-max)

### 13.2 长上下文与架构

- [Qwen2.5-1M Technical Report](https://arxiv.org/abs/2501.15383)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- [Training-Free Long-Context Scaling of Large Language Models / DCA](https://arxiv.org/abs/2402.17463)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

### 13.3 文本专项与视觉文本理解

- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
- [Qwen2.5-Math Technical Report](https://arxiv.org/abs/2409.12122)
- [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176)
- [Qwen3-Reranker-0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Qwen3-Coder 官方博客](https://qwenlm.github.io/blog/qwen3-coder/)
- [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [Qwen-AgentWorld: Language World Models for General Agents](https://arxiv.org/abs/2606.24597)
- [Qwen-UI-Agent Technical Report](https://arxiv.org/abs/2607.28227)
- [Qwen-CUA: Native Computer Use for (almost) Everything](https://arxiv.org/abs/2608.02352)

> 完整技术报告检索记录、证据等级和写作边界见 `sources/research_qwen_technical_reports_20260728.md`；证据化大纲见 `sources/qwen_evidence_outline_20260728.md`；Qwen3.6/Qwen3.7 的最新增量检索、arXiv/OpenAlex 查询与开放状态见 `sources/research_qwen_latest_20260809.md`。

### 13.4 多模态、代码、Agent 与部署补充来源

- [Qwen-VL 官方博客](https://qwenlm.github.io/blog/qwen-vl/)
- [Qwen2-VL 官方博客](https://qwenlm.github.io/blog/qwen2-vl/)
- [Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191)
- [Qwen2.5-VL 官方博客](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [Qwen2-Audio 官方博客](https://qwenlm.github.io/blog/qwen2-audio/)
- [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)
- [Qwen2.5-Omni 官方博客](https://qwenlm.github.io/blog/qwen2.5-omni/)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- [Qwen2.5-Coder-32B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [Qwen-Agent 官方仓库](https://github.com/QwenLM/Qwen-Agent)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)

# 待更新事项

1. Qwen3-Next 目前主要依据官方模型卡，待官方发布完整技术报告后补充优化器、数据配比、并行策略、负载均衡与 MTP 训练权重。
2. Qwen3.6/3.7 的公开信息主要来自仓库、模型卡和 API 产品页；若发布正式技术报告，应重新核对架构、训练数据和 benchmark。
3. Qwen2-Audio、CodeQwen1.5 和 Qwen2.5-Omni 的部分训练超参数、数据许可与完整并行配置未公开，本文已标为“官方未完全公开”。
4. 本次 arXiv export API 调用因运行时网络审批异常未成功，论文正文通过 arXiv HTML 精读；API 原始 XML 待环境恢复后补充。
5. 所有内部/官方 leaderboard 分数仍需在目标硬件、量化、模板、上下文长度与业务数据上复测，不作为生产 SLA。

# 版本记录

| 日期 | 版本 | 更新内容 |
|---|---|---|
| 2026-08-16 | v2.0 | 从 `补充：Qwen.md` 生成规范目标文件 `Qwen.md`；新增更新时间、来源清单、目录入口、证据化七要素模型卡；系统补齐 Qwen-VL/Qwen2-VL/Qwen2.5-VL/Qwen3-VL、Qwen2-Audio、Qwen2.5-Omni/Qwen3-Omni、CodeQwen/Qwen2.5-Coder、QwQ/Qwen3 Thinking、Qwen-Agent、量化和 vLLM；明确 DCA 在 Qwen2 已引入。 |
| 2026-08-09 | v1.x | 原《补充：Qwen.md》内容：Qwen 主线、Qwen3.5/3.6/3.7 增量、长上下文、MoE 与部署。 |
