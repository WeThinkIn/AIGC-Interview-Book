# 目录

## 第一章 Agent 设计模式总览

[1. 为什么 AI Agent 需要设计模式？](#q-001)
  - [面试问题：Agent Pattern、Prompt Pattern、Workflow Pattern 有什么区别？](#q-002)
  - [面试问题：Agent 和 Workflow 应该如何取舍？](#q-003)
  - [面试问题：为什么很多 Agent 项目失败在“过度自主”？](#q-004)

## 第二章 ReAct、CoT 与工具增强推理

[2. ReAct 的核心思想是什么？](#q-005)
  - [面试问题：ReAct 的完整运行循环是什么？](#q-006)
  - [面试问题：ReAct 相比 CoT 的优势和局限是什么？](#q-007)
  - [面试问题：ReAct 与 CoT-SC 混合策略为什么有效？](#q-008)
  - [面试问题：如何避免 ReAct 陷入无效搜索和循环调用？](#q-009)

## 第三章 规划、执行与反思

[3. Plan-and-Solve / Plan-and-Execute 解决什么问题？](#q-010)
  - [面试问题：Plan-and-Solve 与 Plan-and-Execute 有什么区别？](#q-011)
  - [面试问题：ReWOO 为什么要把推理规划和工具观察解耦？](#q-012)
  - [面试问题：Reflection / Reflexion 如何提升 Agent 可靠性？](#q-013)
  - [面试问题：什么时候反思会带来负收益？](#q-014)

## 第四章 搜索式推理与复杂任务求解

[4. Tree-of-Thought、Graph-of-Thought、LATS 分别解决什么问题？](#q-015)
  - [面试问题：Tree-of-Thought 和普通 CoT 的区别是什么？](#q-016)
  - [面试问题：Graph-of-Thought 为什么比树结构更灵活？](#q-017)
  - [面试问题：LATS / MCTS-style Agent 的核心思想是什么？](#q-018)
  - [面试问题：搜索式 Agent 为什么难以直接用于高并发生产场景？](#q-019)

## 第五章 多 Agent 与工作流编排

[5. 多 Agent 系统常见编排模式有哪些？](#q-020)
  - [面试问题：Supervisor、Router、Swarm、Debate 如何对比？](#q-021)
  - [面试问题：Crew / Team 型 Agent 适合什么场景？](#q-022)
  - [面试问题：为什么多 Agent 不一定比单 Agent 更好？](#q-023)
  - [面试问题：如何设计可持久化、可恢复的 Agent 工作流？](#q-024)

---

<h1 id="q-001">1. 为什么 AI Agent 需要设计模式？</h1>

AI Agent 不是简单地“给大模型接几个工具”。一旦任务变长、工具变多、风险变高，系统就会遇到一批重复问题：

- 如何拆解任务。
- 如何选择工具。
- 如何处理工具失败。
- 如何验证中间结果。
- 如何让模型从错误中恢复。
- 如何避免上下文爆炸。
- 如何在人类审批和自动执行之间平衡。
- 如何让多个 Agent 协作而不是互相干扰。

设计模式的价值，就是把这些高频问题沉淀成可复用的结构。常见模式包括：

- ReAct：推理与行动交替。
- Plan-and-Solve：先规划，再求解。
- Plan-and-Execute：规划器与执行器分离。
- ReWOO：先生成带变量依赖的工具计划，再批量执行观察。
- Reflection / Reflexion：执行后自我检查、记录经验、修正策略。
- Tree-of-Thought：对多个推理分支进行搜索和评估。
- Graph-of-Thought：把思考单元组织成图结构。
- LATS：把语言模型推理、行动和搜索树结合。
- Supervisor / Router：由上层协调者分配任务。
- Workflow：用确定性状态机约束 Agent 行为。

面试中要强调：设计模式不是为了炫技，而是为了控制复杂度、降低失败率、提高可解释性和工程可维护性。

<h2 id="q-002">面试问题：Agent Pattern、Prompt Pattern、Workflow Pattern 有什么区别？</h2>

| 类型 | 关注点 | 典型例子 | 风险 |
| --- | --- | --- | --- |
| Prompt Pattern | 如何引导模型输出更好的推理或格式 | CoT、Plan-and-Solve、Self-Consistency | 依赖模型遵循指令，稳定性有限 |
| Agent Pattern | 模型如何循环感知、推理、行动、反思 | ReAct、Reflection、Tool Use、Memory | 工具副作用、上下文漂移、成本增加 |
| Workflow Pattern | 系统如何用确定性流程编排模型和工具 | Router、State Machine、Human Approval、DAG | 灵活性下降，流程设计成本高 |

三者不是互斥关系。一个成熟 Agent 系统通常会同时使用：

- Prompt Pattern 提升单次模型推理质量。
- Agent Pattern 让模型具备多步行动能力。
- Workflow Pattern 用工程边界约束模型行为。

例如，一个代码修复 Agent 可以使用 Plan-and-Solve 生成计划，用 ReAct 循环读取文件和运行测试，再用 Workflow 强制经过“生成 diff -> 运行测试 -> 人工 review -> 创建 PR”的流程。

<h2 id="q-003">面试问题：Agent 和 Workflow 应该如何取舍？</h2>

可以从任务确定性、风险和反馈方式来判断。

| 场景 | 更适合 Agent | 更适合 Workflow |
| --- | --- | --- |
| 任务边界 | 模糊、需要探索 | 明确、步骤稳定 |
| 工具选择 | 需要动态判断 | 工具固定 |
| 结果验证 | 需要多轮信息收集 | 有清晰规则 |
| 风险等级 | 低到中，可人工兜底 | 高风险、强合规 |
| 用户期望 | 希望系统自主推进 | 希望系统可预测 |

经验原则：

- 可枚举、可审计、强合规的任务优先 Workflow。
- 需要探索、推理、动态选择工具的任务适合 Agent。
- 生产系统里常用混合模式：Workflow 管大边界，Agent 处理局部不确定性。

例如报销审批、身份认证、支付转账更适合 Workflow；代码排障、研究调研、复杂数据分析更适合 Agent；企业客服常用 Workflow 控制关键节点，再让 Agent 负责理解用户意图和生成回复。

<h2 id="q-004">面试问题：为什么很多 Agent 项目失败在“过度自主”？</h2>

过度自主指系统给 Agent 太多目标、太多工具和太少约束，期望它自动完成复杂任务。常见后果包括：

- 计划过大，无法收敛。
- 工具调用过多，成本和延迟不可控。
- 中间结果没有验证，错误逐步放大。
- 上下文越来越长，任务目标逐渐漂移。
- 权限过宽，容易产生真实副作用。
- 没有人工审批，失败后难以追责。

优化思路：

- 把大任务切成可验证的小任务。
- 让 Agent 每一步产生结构化状态。
- 对高风险工具加入审批。
- 用 Workflow 限制关键路径。
- 设置最大步数、预算和超时。
- 在关键节点引入测试、评估器或人工确认。

一句话：生产级 Agent 的核心不是“让模型想做什么就做什么”，而是“让模型在明确边界内自主解决局部不确定性”。

---

<h1 id="q-005">2. ReAct 的核心思想是什么？</h1>

ReAct 是 Reasoning and Acting 的缩写，核心思想是让模型在解决任务时交替进行推理和行动：

$$
\text{Thought} \rightarrow \text{Action} \rightarrow \text{Observation} \rightarrow \text{Thought} \rightarrow \cdots \rightarrow \text{Final Answer}
$$

它解决了纯 CoT 和纯工具调用各自的缺陷：

- 纯 CoT 只能依赖模型内部知识，容易幻觉。
- 纯工具调用缺少高层推理，容易盲目搜索。
- ReAct 用推理决定下一步行动，再用外部观察修正推理。

ReAct 适合：

- 知识密集型问答。
- 多跳检索。
- 浏览器任务。
- 交互式环境决策。
- 工具调用较少但需要动态判断的任务。
- 编码 Agent 的探索和验证阶段。

<h2 id="q-006">面试问题：ReAct 的完整运行循环是什么？</h2>

标准 ReAct 循环包含五个阶段：

1. **任务理解**

   解析用户目标、约束、可用工具和停止条件。

2. **Thought**

   生成当前推理：已经知道什么、缺什么、下一步该查什么或做什么。

3. **Action**

   选择工具并给出结构化参数，例如搜索、读取文件、执行命令、调用 API。

4. **Observation**

   工具返回结果。系统需要把结果以安全、简洁、可引用的方式放回上下文。

5. **Final Answer**

   当信息足够或达到停止条件时，模型输出最终答案。

工程实现时通常不会把所有内部推理完整暴露给用户，而是保存为 trace 或压缩成可解释摘要。对于高风险场景，Action 前还需要权限审批或策略检查。

<h2 id="q-007">面试问题：ReAct 相比 CoT 的优势和局限是什么？</h2>

| 维度 | CoT | ReAct |
| --- | --- | --- |
| 信息来源 | 模型内部知识 | 内部推理 + 外部工具观察 |
| 事实性 | 容易受训练数据影响 | 可通过检索和工具降低幻觉 |
| 成本 | 低 | 工具调用和多轮交互成本更高 |
| 可解释性 | 有推理链但可能不接地 | 能看到行动和观察轨迹 |
| 稳定性 | 对简单推理稳定 | 可能陷入工具循环 |
| 适用任务 | 数学、逻辑、常识推理 | 检索、交互、排障、动态任务 |

ReAct 的局限：

- 工具结果质量会强烈影响最终答案。
- 搜索关键词错误会导致偏航。
- 贪心决策可能反复调用无效工具。
- 每轮工具调用都增加延迟和成本。
- 工具返回内容可能带有 prompt injection。
- 长任务中 observation 太多会挤爆上下文。

所以 ReAct 不是万能模式。对于步骤明确、无需外部信息的问题，CoT 或 Plan-and-Solve 可能更便宜、更稳定。

<h2 id="q-008">面试问题：ReAct 与 CoT-SC 混合策略为什么有效？</h2>

CoT-SC 指 Chain-of-Thought Self-Consistency，即让模型生成多个推理路径，再通过投票或一致性选择答案。它利用模型内部知识，适合低工具成本的推理任务。

ReAct 与 CoT-SC 可以互补：

1. **ReAct 失败后回退 CoT-SC**

   当搜索失败、工具不可用或环境噪声大时，可以让模型基于内部知识生成多个候选推理，再选择最一致的答案。

2. **CoT-SC 置信度低时切换 ReAct**

   当多个候选答案分歧很大，说明内部知识不稳定，应主动调用外部工具验证。

3. **先 ReAct 收集事实，再 CoT-SC 推理**

   对需要事实依据又需要复杂推理的问题，可以先用工具拿证据，再对证据进行多路径推理。

4. **先 CoT 拆解问题，再 ReAct 查关键缺口**

   对复杂研究任务，可以先拆成子问题，只对缺失信息调用工具。

面试中可以总结：ReAct 提供外部 grounding，CoT-SC 提供内部多样性和一致性，两者组合能在事实性、效率和鲁棒性之间做动态平衡。

<h2 id="q-009">面试问题：如何避免 ReAct 陷入无效搜索和循环调用？</h2>

常见策略：

- 设置最大工具调用步数。
- 对连续相同工具和相似参数做去重。
- 要求每次 Action 前说明“本次工具调用将获得什么新增信息”。
- 工具失败后必须改变查询策略，而不是重复同一查询。
- 对 observation 做摘要，避免上下文污染。
- 引入 evaluator 判断当前信息是否足够。
- 对低收益工具调用设置成本惩罚。
- 当多次失败时切换到 Plan-and-Solve、CoT-SC 或人工求助。

工程上可以维护一个 action history：

| 字段 | 作用 |
| --- | --- |
| tool | 记录调用了什么工具 |
| arguments | 判断是否重复调用 |
| expected_gain | 记录本次期望获得的信息 |
| observation_summary | 压缩工具结果 |
| success / failure | 供后续策略调整 |

如果 Agent 连续多轮没有获得新信息，应触发 stop、replan 或 ask-human，而不是继续消耗 token。

---

<h1 id="q-010">3. Plan-and-Solve / Plan-and-Execute 解决什么问题？</h1>

Plan-and-Solve 和 Plan-and-Execute 都强调“先规划，再执行”，目标是减少模型直接作答时常见的三类错误：

- 步骤缺失。
- 计算错误。
- 任务理解偏差。

它们适合：

- 数学推理。
- 多步骤数据分析。
- 代码修改任务。
- 研究报告生成。
- 企业流程执行。
- 需要中间状态可检查的任务。

核心思想可以表示为：

$$
\text{Task} \rightarrow \text{Plan} \rightarrow \text{Step Execution} \rightarrow \text{Verification} \rightarrow \text{Answer}
$$

<h2 id="q-011">面试问题：Plan-and-Solve 与 Plan-and-Execute 有什么区别？</h2>

| 维度 | Plan-and-Solve | Plan-and-Execute |
| --- | --- | --- |
| 本质 | Prompting 方法 | Agent 架构模式 |
| 规划者 | 同一个模型在答案前先写计划 | 可由独立 planner 生成计划 |
| 执行者 | 通常还是同一模型完成推理 | executor 可调用工具、子 Agent 或工作流 |
| 状态管理 | 较轻量 | 需要任务状态、进度、失败恢复 |
| 适用场景 | 数学、逻辑、单轮复杂问答 | 编码、研究、自动化、多工具任务 |

Plan-and-Solve 更偏提示工程，常见形式是要求模型先列计划，再按计划求解。Plan-and-Execute 更偏工程系统，通常包含：

- Planner：生成可执行步骤。
- Executor：逐步执行计划。
- State Store：记录进度。
- Evaluator：检查每一步是否成功。
- Replanner：失败时调整计划。

面试中可以说：Plan-and-Solve 是“让一个模型先想清楚再回答”，Plan-and-Execute 是“把规划和执行做成可持久化、可观察、可恢复的系统”。

<h2 id="q-012">面试问题：ReWOO 为什么要把推理规划和工具观察解耦？</h2>

ReWOO 的思想是 Decoupling Reasoning from Observations。传统 ReAct 每一步都要等待工具 observation 后再继续推理，导致：

- 模型调用次数多。
- 工具调用串行，延迟高。
- 观察结果会不断干扰后续推理。
- 长任务中上下文迅速增长。

ReWOO 会先生成一个带变量依赖的计划，例如：

```text
Plan: 需要比较 A 公司和 B 公司最近财报表现。
E1 = Search[A 公司 2025 年报 营收 利润]
E2 = Search[B 公司 2025 年报 营收 利润]
E3 = Compare[E1, E2]
Final = Summarize[E3]
```

这样做的好处：

- 可提前看出工具依赖关系。
- 无依赖的工具可以并行执行。
- 执行器不需要每一步都调用大模型。
- 规划阶段更稳定，观察阶段更可控。
- 更容易审计和缓存工具结果。

局限是：如果初始计划错了，后续执行会整体偏离；如果任务高度交互、每一步都依赖环境反馈，ReWOO 不如 ReAct 灵活。

<h2 id="q-013">面试问题：Reflection / Reflexion 如何提升 Agent 可靠性？</h2>

Reflection 强调模型对自己的输出或行动进行检查和修正。Reflexion 更进一步，把失败经验写入记忆，用于后续尝试。

常见流程：

1. Agent 执行任务。
2. Evaluator 判断结果是否成功。
3. 如果失败，Reflector 总结失败原因。
4. 将经验写入短期或长期记忆。
5. 下一轮执行时读取经验，调整策略。

适合场景：

- 代码生成和测试修复。
- 浏览器自动化。
- 问答自检。
- 复杂工具调用。
- 有明确成功/失败信号的任务。

一个好的 reflection 不应该只是“我应该更小心”，而应该具体到：

- 哪一步错了。
- 错误证据是什么。
- 下次不要重复什么。
- 应该尝试哪个新策略。
- 是否需要额外工具或信息。

例如代码 Agent 测试失败后，反思应记录“测试失败是因为 mock 数据缺少 user_id 字段，下一轮先检查 fixture，而不是继续改业务逻辑”。

<h2 id="q-014">面试问题：什么时候反思会带来负收益？</h2>

Reflection 不是越多越好。负收益常见于：

- 没有可靠 evaluator，模型只能自说自话。
- 反思内容空泛，不能改变下一步行动。
- 任务很简单，反思增加不必要成本。
- 模型把错误反思写入长期记忆，污染后续任务。
- 多轮反思导致过度修正，偏离原始需求。
- 反思暴露过多中间推理，增加安全和隐私风险。

优化建议：

- 只有失败、低置信度或高风险任务才触发反思。
- 反思必须结构化：失败原因、证据、修正策略、下步行动。
- 反思写入长期记忆前要经过质量过滤。
- 对同一问题设置最大反思轮数。
- 反思要结合外部反馈，例如测试结果、评测器、用户确认。

面试中可以说：Reflection 的价值依赖反馈信号。如果没有外部验证，它很容易从“纠错机制”退化成“更长的幻觉”。

---

<h1 id="q-015">4. Tree-of-Thought、Graph-of-Thought、LATS 分别解决什么问题？</h1>

复杂任务往往不是一条线性推理链就能解决。模型可能需要探索多个候选方案、比较分支质量、回溯失败路径。搜索式推理模式就是为这类问题设计的。

三类代表模式：

| 模式 | 核心结构 | 适合任务 |
| --- | --- | --- |
| Tree-of-Thought | 多个 thought 组成搜索树 | 谜题、规划、复杂推理 |
| Graph-of-Thought | thought 组成可合并、可回环的图 | 多来源信息整合、复杂依赖 |
| LATS | 语言模型 + 行动 + 树搜索 + 价值评估 | 交互式任务、需要探索和试错的环境 |

这些模式的共同点是：不再接受模型第一条推理路径，而是生成多个候选路径，并通过评估、搜索、回溯选择更优路径。

<h2 id="q-016">面试问题：Tree-of-Thought 和普通 CoT 的区别是什么？</h2>

普通 CoT 是单路径推理：

$$
\text{Thought}_1 \rightarrow \text{Thought}_2 \rightarrow \text{Thought}_3 \rightarrow \text{Answer}
$$

Tree-of-Thought 是多分支搜索：

$$
\text{State} \rightarrow \{\text{Thought}_1, \text{Thought}_2, \text{Thought}_3\} \rightarrow \text{Evaluate} \rightarrow \text{Search}
$$

关键差异：

- CoT 一次生成一条链。
- ToT 在每个状态生成多个候选 thought。
- ToT 会评估候选 thought 的价值。
- ToT 可以使用 BFS、DFS、Beam Search 等搜索策略。
- ToT 可以回溯，不会被第一条错误路径锁死。

ToT 适合解题空间较大、需要探索的任务，例如数独、24 点游戏、复杂规划。但它成本高，不适合简单问答或强实时场景。

<h2 id="q-017">面试问题：Graph-of-Thought 为什么比树结构更灵活？</h2>

树结构假设每个 thought 只有一个父节点，分支之间相对独立。但真实复杂任务中，经常需要：

- 合并两个分支的信息。
- 对同一个中间结论反复修正。
- 从多个证据共同支持一个判断。
- 在图中形成依赖关系，而不是简单层级。

Graph-of-Thought 把 thought 看成图节点，把推理关系看成边。这样可以支持：

- aggregation：聚合多个 thought。
- refinement：迭代改进某个 thought。
- branching：生成多个候选方向。
- scoring：对节点或路径打分。
- pruning：删除低质量分支。

适合场景：

- 研究综述。
- 多文档问答。
- 复杂决策分析。
- 多 Agent 辩论结果聚合。
- 代码架构方案比较。

代价是工程复杂度更高，需要维护图状态、依赖关系、节点评分和终止条件。

<h2 id="q-018">面试问题：LATS / MCTS-style Agent 的核心思想是什么？</h2>

LATS 可以理解为把语言模型 Agent 和树搜索结合起来。它通常包含：

- 生成候选行动。
- 执行动作并获得环境反馈。
- 对状态进行价值评估。
- 根据搜索策略选择下一步。
- 从成功或失败轨迹中反思学习。

MCTS-style Agent 的关键不是只让模型“想一步”，而是让系统在多条行动轨迹中搜索更优路径。

典型流程：

1. Selection：选择当前最有潜力的节点。
2. Expansion：扩展多个候选 thought/action。
3. Simulation：模拟或执行后续步骤。
4. Evaluation：评估结果价值。
5. Backpropagation：把价值反馈给上层节点。

适合：

- Web 交互任务。
- 游戏或环境探索。
- 复杂工具链任务。
- 需要试错和回溯的决策。

局限：

- token 和工具调用成本高。
- 需要可靠的 value function。
- 搜索树可能爆炸。
- 对实时性要求高的产品不友好。

<h2 id="q-019">面试问题：搜索式 Agent 为什么难以直接用于高并发生产场景？</h2>

主要原因：

- **成本不可控**：分支越多，模型调用和工具调用越多。
- **延迟高**：搜索需要多轮生成、执行、评估。
- **状态复杂**：需要维护搜索树或图状态。
- **评估困难**：很多任务没有清晰 reward。
- **工具副作用**：不能随便在真实环境里试错。
- **可观测性要求高**：必须记录每个分支为什么被选择或剪枝。

生产优化方式：

- 对高价值、低频复杂任务使用搜索式 Agent。
- 对普通任务使用 ReAct / Workflow。
- 限制搜索宽度和深度。
- 使用小模型做初筛，大模型做关键评估。
- 对工具调用使用模拟环境或只读模式。
- 缓存中间 thought、检索结果和评估结果。

面试中可以说：搜索式推理提升上限，但牺牲成本和延迟；它更适合“难题求优”，不适合“高并发低成本常规请求”。

---

<h1 id="q-020">5. 多 Agent 系统常见编排模式有哪些？</h1>

多 Agent 编排的目标是让多个具有不同角色、工具、记忆或权限的 Agent 协同完成任务。常见模式包括：

- Supervisor：一个上级 Agent 分配任务和汇总结果。
- Router：根据输入类型选择最合适的 Agent。
- Swarm：多个 Agent 自组织协作或交接。
- Debate：多个 Agent 生成不同观点，再由 judge 决策。
- Crew / Team：角色固定的团队协作。
- Pipeline：多个 Agent 按固定顺序处理任务。
- Blackboard：多个 Agent 读写共享工作区。
- Market / Bidding：Agent 根据能力声明竞争任务。

实际工程里，最常见的是 Supervisor、Router、Pipeline 和 Team，因为它们可控、可观测、容易落地。

<h2 id="q-021">面试问题：Supervisor、Router、Swarm、Debate 如何对比？</h2>

| 模式 | 核心思想 | 优点 | 风险 |
| --- | --- | --- | --- |
| Supervisor | 上级 Agent 拆分任务并协调下级 | 控制强，适合复杂项目 | 上级成为瓶颈 |
| Router | 根据任务类型路由到专门 Agent | 简单高效，适合意图分发 | 路由错误会影响结果 |
| Swarm | Agent 之间动态交接和协作 | 灵活，适合开放任务 | 可控性和调试难度高 |
| Debate | 多方提出观点，由评审选择 | 可提升复杂判断质量 | 成本高，可能互相强化错误 |

选择建议：

- 客服和办公自动化优先 Router。
- 研究报告和软件工程任务适合 Supervisor。
- 创意、开放探索可以尝试 Swarm。
- 高价值决策、方案评审可以使用 Debate。

面试中要注意：多 Agent 编排本质是组织结构设计，不是 Agent 数量越多越先进。

<h2 id="q-022">面试问题：Crew / Team 型 Agent 适合什么场景？</h2>

Crew / Team 型 Agent 会给不同 Agent 分配固定角色，例如：

- Planner：制定计划。
- Researcher：检索资料。
- Engineer：实现代码。
- Reviewer：审查质量。
- Tester：运行测试。
- Writer：整理文档。

适合场景：

- 角色边界清晰。
- 任务可拆分。
- 每个角色需要不同工具或提示词。
- 输出需要多轮审查。
- 可以接受更高延迟和成本。

不适合：

- 简单问答。
- 强实时交互。
- 高度耦合、难以拆分的任务。
- 工具权限难以隔离的任务。

工程建议：

- 明确每个 Agent 的输入、输出和停止条件。
- 给每个 Agent 独立上下文，避免互相污染。
- 由 Supervisor 汇总而不是让所有 Agent 共享完整历史。
- 对写文件、发消息、执行命令等副作用操作设置单一责任人。

<h2 id="q-023">面试问题：为什么多 Agent 不一定比单 Agent 更好？</h2>

多 Agent 会带来额外复杂度：

- 通信成本增加。
- 上下文重复和信息丢失。
- 角色之间可能相互矛盾。
- 多个 Agent 可能争抢同一工具或文件。
- 错误会跨 Agent 传播。
- 调试难度大幅上升。
- token 和延迟成本更高。

什么时候单 Agent 更好？

- 任务短。
- 工具少。
- 目标明确。
- 不需要角色分工。
- 用户希望快速得到结果。

什么时候多 Agent 值得？

- 任务天然可拆。
- 需要并行探索。
- 需要互相审查。
- 不同子任务需要不同权限。
- 需要专家角色分工。

一句话：多 Agent 不是能力放大器的免费午餐，它更像组织管理。组织结构设计不好，人多反而更乱。

<h2 id="q-024">面试问题：如何设计可持久化、可恢复的 Agent 工作流？</h2>

可持久化工作流的核心是把 Agent 运行过程从“临时对话”变成“状态机”。

关键设计：

1. **显式状态**

   记录任务目标、当前步骤、已完成步骤、失败原因、待审批项、工具结果摘要。

2. **节点化执行**

   将流程拆成 planner、tool_call、evaluate、human_review、replan、finalize 等节点。

3. **持久化存储**

   每个节点完成后保存状态，支持进程重启、断线恢复和历史追溯。

4. **可重试和幂等**

   工具调用要区分只读和副作用。副作用操作需要 idempotency key。

5. **Human-in-the-loop**

   高风险节点进入人工审批，用户可以 approve、reject、edit、abort。

6. **可观测性**

   保存模型输入输出、工具调用、错误、成本、耗时和最终结果。

7. **回滚策略**

   对文件修改、数据库写入、外部 API 调用设计补偿或人工恢复流程。

典型状态流：

$$
\text{intake} \rightarrow \text{plan} \rightarrow \text{execute} \rightarrow \text{evaluate} \rightarrow \text{replan / approve} \rightarrow \text{finalize}
$$

面试中可以把 LangGraph 这类框架作为例子：它的价值不是“让 Agent 更聪明”，而是把复杂 Agent 编排成可持久化、可中断、可恢复、可观测的图状态机。

---

## 高频速记

1. Agent 设计模式用于控制复杂度，而不是让系统显得更花哨。
2. Prompt Pattern 优化单次推理，Agent Pattern 组织多步行动，Workflow Pattern 约束工程流程。
3. 生产系统通常是 Workflow 管边界，Agent 解决局部不确定性。
4. ReAct 通过 Thought / Action / Observation 循环把推理和工具连接起来。
5. ReAct 适合动态信息获取，但容易陷入无效搜索和工具循环。
6. CoT-SC 提供内部多路径一致性，ReAct 提供外部 grounding，二者可以互补。
7. Plan-and-Solve 更偏 prompting，Plan-and-Execute 更偏工程架构。
8. ReWOO 通过先规划工具依赖减少串行模型调用，适合可提前拆解的任务。
9. Reflection 依赖可靠反馈信号，没有外部验证时可能放大幻觉。
10. Tree-of-Thought 和 Graph-of-Thought 提升复杂推理上限，但成本和延迟较高。
11. LATS / MCTS-style Agent 适合需要搜索、试错和回溯的交互环境。
12. 多 Agent 是组织结构设计，不是 Agent 数量越多越好。
13. Supervisor 适合复杂任务协调，Router 适合意图分发，Debate 适合高价值判断。
14. 可持久化 Agent 工作流需要状态机、节点、持久化、审批、重试和可观测性。

## 参考资料

- Yao et al., [**ReAct: Synergizing Reasoning and Acting in Language Models**](https://arxiv.org/abs/2210.03629), 2022.
- Wang et al., [**Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**](https://arxiv.org/abs/2305.04091), 2023.
- Xu et al., [**ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models**](https://arxiv.org/abs/2305.18323), 2023.
- Shinn et al., [**Reflexion: Language Agents with Verbal Reinforcement Learning**](https://arxiv.org/abs/2303.11366), 2023.
- Yao et al., [**Tree of Thoughts: Deliberate Problem Solving with Large Language Models**](https://arxiv.org/abs/2305.10601), 2023.
- Besta et al., [**Graph of Thoughts: Solving Elaborate Problems with Large Language Models**](https://arxiv.org/abs/2308.09687), 2023.
- Zhou et al., [**Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models**](https://arxiv.org/abs/2310.04406), 2023.
- LangChain, [**LangGraph Documentation**](https://docs.langchain.com/oss/python/langgraph/overview).
