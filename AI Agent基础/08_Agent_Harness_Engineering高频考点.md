# 目录

## 第一章 Harness Engineering 总览

[1. AI Agent 中的 Harness Engineering 是什么？](#q-001)
  - [面试问题：Harness、Benchmark、Eval、AgentOps 有什么区别？](#q-002)
  - [面试问题：为什么 Agent 比普通 LLM 更需要 Harness Engineering？](#q-003)
  - [面试问题：一个完整 Agent Harness 通常包含哪些模块？](#q-004)

## 第二章 任务、环境与运行器

[2. Agent Harness 的任务数据应该如何设计？](#q-005)
  - [面试问题：Task Spec、Environment Spec、Expected Outcome 应该包含什么？](#q-006)
  - [面试问题：Environment Simulator / Sandbox 的核心价值是什么？](#q-007)
  - [面试问题：Agent Runner 如何保证可复现、可对比、可回放？](#q-008)

## 第三章 Trace、Grader 与指标体系

[3. Agent Harness 为什么必须记录完整 Trace？](#q-009)
  - [面试问题：Outcome Grading、Trajectory Grading、State Grading 如何区分？](#q-010)
  - [面试问题：规则判分、单元测试、LLM-as-Judge、人审如何组合？](#q-011)
  - [面试问题：Agent Harness 应该输出哪些核心指标？](#q-012)

## 第四章 主流场景 Harness 设计

[4. 编码 Agent Harness 应该如何设计？](#q-013)
  - [面试问题：SWE-bench Harness 给编码 Agent 评测带来哪些启发？](#q-014)
  - [面试问题：浏览器 / GUI Agent Harness 应该如何设计？](#q-015)
  - [面试问题：客服 / 工具调用 Agent Harness 应该如何设计？](#q-016)
  - [面试问题：多 Agent / A2A Harness 应该如何设计？](#q-017)

## 第五章 回归门禁、CI/CD 与生产闭环

[5. Agent Harness 如何接入 CI/CD 和发布门禁？](#q-018)
  - [面试问题：Smoke Eval、Regression Eval、Safety Eval、Canary Eval 如何分层？](#q-019)
  - [面试问题：如何用 Harness 做失败样本回流和持续优化？](#q-020)
  - [面试问题：Harness 如何服务线上 Shadow Mode 和灰度发布？](#q-021)

## 第六章 前沿趋势与高频架构题

[6. 2026 年 Agent Harness Engineering 有哪些主流趋势？](#q-022)
  - [面试问题：Model-native Harness 和传统脚本 Harness 有什么区别？](#q-023)
  - [面试问题：如何回答“请设计一个企业级 Agent Harness 平台”？](#q-024)

---

<h1 id="q-001">1. AI Agent 中的 Harness Engineering 是什么？</h1>

Harness Engineering 指围绕 Agent 构建一套可复现、可运行、可评估、可回放、可上线门禁的工程系统。它不只是“写几个测试用例”，而是把 Agent 放进接近真实业务的受控环境里，让它真正执行任务，然后系统性记录和评估结果。

可以把它理解成：

$$
\text{Agent Harness} =
\text{Tasks} + \text{Environment} + \text{Runner} + \text{Trace} + \text{Grader} + \text{Report} + \text{Gate}
$$

Agent Harness 的目标是回答五个问题：

- Agent 能否完成任务？
- Agent 是如何完成或失败的？
- 新版本相比旧版本是否退化？
- 失败是模型问题、工具问题、上下文问题、权限问题还是环境问题？
- 当前版本是否可以上线、灰度或扩大权限？

**面试金句：**

Harness Engineering 是 Agent 从 demo 走向生产的质量基础设施。没有 Harness，就只能靠人工试用和主观感觉判断 Agent 是否变好。

<h2 id="q-002">面试问题：Harness、Benchmark、Eval、AgentOps 有什么区别？</h2>

| 概念 | 核心含义 | 例子 |
| --- | --- | --- |
| Benchmark | 标准化任务集和指标 | SWE-bench、WebArena、OSWorld、GAIA、tau-bench |
| Eval | 一次评估过程或评估配置 | 在某个数据集上评估新 prompt、新模型、新工具 |
| Harness | 执行 eval 的工程框架 | 准备环境、运行 Agent、收集 trace、判分、生成报告 |
| AgentOps | Agent 上线后的运营治理体系 | tracing、监控、灰度、回滚、事故响应、失败回流 |

关系可以这样理解：

- Benchmark 提供“考卷”。
- Eval 定义“这次考试怎么考”。
- Harness 提供“考场、监考、阅卷和成绩单”。
- AgentOps 把成绩用于“发布、灰度、回滚和持续改进”。

面试中要避免把 Benchmark 等同于 Harness。公开 Benchmark 只是外部参考；企业真正需要的是能跑自家工具、自家流程、自家权限和自家数据的 Harness。

<h2 id="q-003">面试问题：为什么 Agent 比普通 LLM 更需要 Harness Engineering？</h2>

普通 LLM 评测通常关注“输入 -> 输出”是否正确。Agent 评测更难，因为 Agent 有工具调用、环境状态、长任务、多轮交互、权限审批和真实副作用。

Agent Harness 必须覆盖：

- **Outcome**：最终任务是否完成。
- **Trajectory**：中间步骤是否合理。
- **Tool Use**：是否选对工具、参数是否正确。
- **State Change**：数据库、文件、工单、页面状态是否符合预期。
- **Safety**：是否越权、泄密、误执行高风险动作。
- **Cost / Latency**：是否在预算内完成。
- **Robustness**：多次运行是否稳定。
- **Recoverability**：失败后是否可重试、可回滚、可恢复。

例如一个客服 Agent 最终回答看似正确，但如果它在中间调用了越权订单查询工具，或者错误修改了数据库状态，这就是严重失败。只看最终文本会漏掉这些风险。

<h2 id="q-004">面试问题：一个完整 Agent Harness 通常包含哪些模块？</h2>

完整 Agent Harness 通常包含九个模块：

1. **Task Dataset**

   任务输入、初始状态、用户目标、权限范围、风险标签和期望结果。

2. **Environment Builder**

   构造仓库、网页、数据库、文件系统、API mock、用户模拟器或沙箱。

3. **Agent Runner**

   以固定模型、prompt、工具、策略、随机种子和预算运行 Agent。

4. **Tool / API Simulator**

   模拟真实工具返回、异常、延迟、权限错误和边界情况。

5. **Trace Collector**

   记录模型调用、工具调用、状态变化、审批事件、成本和耗时。

6. **Evaluator / Grader**

   评估最终状态、最终答案、执行轨迹、安全行为和成本。

7. **Report Dashboard**

   展示通过率、失败分类、版本对比、趋势和 trace 链接。

8. **Regression Gate**

   接入 CI/CD，新版本退化时阻止发布。

9. **Failure Feedback Loop**

   将线上失败转成新任务，补充到回归集和安全集。

工程上可以简化为四层：

$$
\text{Dataset} \rightarrow \text{Run} \rightarrow \text{Judge} \rightarrow \text{Gate}
$$

---

<h1 id="q-005">2. Agent Harness 的任务数据应该如何设计？</h1>

Agent Harness 的任务数据不是简单的 prompt 列表，而是可执行任务规格。一个高质量任务样本应该能让不同 Agent 在同样条件下运行，并被同样规则评估。

推荐任务结构：

```yaml
task_id: customer_refund_001
domain: retail
difficulty: medium
risk_level: high
input:
  user_message: "我想退掉昨天买的蓝色外套"
initial_state:
  database_snapshot: snapshots/retail_001.sql
  policy_doc: policies/refund_policy.md
tools:
  allowed:
    - search_order
    - inspect_item
    - create_refund_request
  denied:
    - direct_refund_payment
expected:
  final_response_contains:
    - "需要确认订单"
  final_database_state:
    refund_status: "pending_review"
  forbidden_actions:
    - "直接退款"
grading:
  outcome_weight: 0.5
  safety_weight: 0.3
  trajectory_weight: 0.2
```

关键是把任务从“自然语言请求”升级成“输入 + 环境 + 约束 + 期望状态 + 判分规则”。

<h2 id="q-006">面试问题：Task Spec、Environment Spec、Expected Outcome 应该包含什么？</h2>

**Task Spec** 描述任务本身：

- task_id。
- 用户输入。
- 任务类型。
- 难度。
- 风险等级。
- 允许工具。
- 禁止动作。
- 最大轮数、最大耗时、最大成本。
- 是否需要人类确认。

**Environment Spec** 描述运行环境：

- 初始数据库快照。
- 初始文件系统。
- 仓库 commit。
- 浏览器页面状态。
- mock API。
- 用户模拟器 persona。
- 网络、权限和密钥配置。
- 随机种子。

**Expected Outcome** 描述应该发生什么：

- 最终文本答案。
- 最终数据库状态。
- 文件 diff。
- 通过的测试。
- 生成的 artifact。
- 不应该发生的动作。
- 必须引用的证据。
- 允许的多种合法路径。

高质量 Harness 不应只依赖“最终答案字符串匹配”。很多 Agent 任务存在多条合理路径，真正关键的是最终状态、业务规则和安全边界。

<h2 id="q-007">面试问题：Environment Simulator / Sandbox 的核心价值是什么？</h2>

Environment Simulator / Sandbox 的价值是让 Agent 在可控环境中执行真实动作，但不会伤害真实系统。

它解决四个问题：

1. **可复现**

   每次评测从同一个初始状态开始。

2. **可隔离**

   文件、数据库、浏览器、网络和密钥不会污染生产环境。

3. **可注入异常**

   可以模拟 API 超时、权限不足、页面变化、工具失败和用户反悔。

4. **可判分**

   评测结束后可以比较数据库、文件 diff、页面状态或 artifact。

不同场景的环境设计不同：

| 场景 | 环境模拟重点 |
| --- | --- |
| 编码 Agent | Git repo、base commit、依赖环境、测试命令、Docker 沙箱 |
| 浏览器 Agent | 页面 DOM、登录态、表单、按钮、弹窗、网络响应 |
| 客服 Agent | 用户模拟器、订单数据库、政策文档、业务 API |
| 数据分析 Agent | 数据库快照、查询权限、指标口径、报表模板 |
| 多 Agent | Agent Card、消息总线、任务状态、handoff artifact |

<h2 id="q-008">面试问题：Agent Runner 如何保证可复现、可对比、可回放？</h2>

Agent Runner 是 Harness 的执行核心。它要保证同一个任务在不同版本上尽量公平可比。

关键设计：

- 固定 model / prompt / tool / policy / memory / RAG index 版本。
- 固定温度、top_p、最大轮数、最大工具调用数。
- 固定环境快照和随机种子。
- 每个任务独立 sandbox。
- 每次运行分配 run_id。
- 记录完整版本组合。
- 工具调用结果可缓存或可重放。
- 支持 timeout、cancel、retry 和 cleanup。
- 支持批量并发，但不能让任务之间共享脏状态。

Runner 输出至少包含：

- run_id。
- task_id。
- agent_version。
- final_status。
- trace_id。
- cost。
- latency。
- score。
- failure_tags。
- artifacts。

面试中可以说：Runner 的核心不是“把 Agent 调起来”，而是让每次运行成为可复现的实验。

---

<h1 id="q-009">3. Agent Harness 为什么必须记录完整 Trace？</h1>

Agent 的成败往往藏在中间步骤里。完整 Trace 是 Harness 的事实依据。

Trace 应记录：

- 输入消息。
- system / developer / policy 版本。
- 模型调用参数。
- 每次模型输出。
- 工具选择和参数。
- 工具返回、错误和耗时。
- 权限审批请求和结果。
- RAG 检索内容。
- Memory 读写。
- handoff / subagent 事件。
- 环境状态变化。
- token、成本和延迟。
- guardrail 命中。
- final answer 和 artifact。

Trace 有四个用途：

- **Debug**：定位失败发生在哪一步。
- **Grading**：评价轨迹是否合规。
- **Audit**：追踪谁批准了什么动作。
- **Dataset Mining**：把失败 trace 转成新评测样本。

OpenAI 的 trace grading 也体现了这个趋势：Agent 评测不再只看最终输出，而是对端到端决策、工具调用和执行轨迹做结构化评分。

<h2 id="q-010">面试问题：Outcome Grading、Trajectory Grading、State Grading 如何区分？</h2>

| Grading 类型 | 评估对象 | 适合场景 |
| --- | --- | --- |
| Outcome Grading | 最终答案或最终 artifact | 问答、报告、代码 patch、客服回复 |
| Trajectory Grading | 中间计划、工具调用、权限请求、观察解释 | 工具型 Agent、多步任务、安全审计 |
| State Grading | 最终环境状态 | 数据库、文件系统、订单状态、页面状态 |
| Safety Grading | 是否越权、泄密、误执行危险操作 | 企业 Agent、浏览器 Agent、编码 Agent |
| Cost Grading | token、耗时、工具次数、人工审批次数 | 生产上线和平台治理 |

例子：

- 编码 Agent：Outcome 看测试是否通过，Trajectory 看是否读了相关文件、是否乱改无关文件，State 看 diff 是否符合预期。
- 客服 Agent：Outcome 看回复是否正确，Trajectory 看是否遵守政策，State 看订单数据库是否变成期望状态。
- 浏览器 Agent：Outcome 看任务是否完成，Trajectory 看是否点击危险按钮前确认，State 看页面或后台状态是否正确。

<h2 id="q-011">面试问题：规则判分、单元测试、LLM-as-Judge、人审如何组合？</h2>

推荐优先级：

1. **确定性规则优先**

   例如数据库字段、文件 diff、API 调用次数、权限是否越界。

2. **单元测试 / 集成测试优先**

   编码 Agent 应尽量用测试结果判定功能是否修复。

3. **LLM-as-Judge 评估开放式质量**

   适合摘要忠实性、回答完整性、沟通质量、轨迹合理性。

4. **人工复核高风险样本**

   金融、医疗、法律、合规、重大权限变更不能完全依赖模型裁判。

组合方式：

| 判分方式 | 优点 | 局限 |
| --- | --- | --- |
| 规则判分 | 稳定、便宜、可解释 | 覆盖不了开放式质量 |
| 单元测试 | 对代码功能强约束 | 测试可能不完整 |
| 状态比较 | 适合业务系统 | 需要设计数据库快照和 diff |
| LLM-as-Judge | 灵活，适合语义评价 | 有偏差，可能被注入影响 |
| 人工复核 | 高可信 | 成本高，难规模化 |

面试高分回答：能用确定性判分的地方不要交给 LLM-as-Judge；LLM-as-Judge 适合补足开放式语义评价，而不是替代安全边界。

<h2 id="q-012">面试问题：Agent Harness 应该输出哪些核心指标？</h2>

指标应覆盖质量、安全、稳定性、成本和工程效率。

| 指标类别 | 典型指标 |
| --- | --- |
| 任务质量 | success rate、pass@1、pass@k、exact match、state match |
| 稳定性 | 多次运行方差、一致性、重试成功率 |
| 工具使用 | 工具调用成功率、参数错误率、无效调用率 |
| 轨迹质量 | 平均步骤数、循环调用率、反思次数、handoff 成功率 |
| 安全 | 越权率、注入攻击成功率、敏感信息泄露率、高风险动作确认率 |
| 成本 | token、费用、工具成本、人工审批次数 |
| 延迟 | 平均耗时、P95/P99、超时率 |
| 回归 | 新版本相比基线的提升、退化、失败类型变化 |

生产中还应关注：

- cost per successful task。
- human takeover rate。
- false allow / false block。
- data leakage incidents。
- rollback rate。
- trace coverage。

---

<h1 id="q-013">4. 编码 Agent Harness 应该如何设计？</h1>

编码 Agent Harness 的目标是评估 Agent 是否能在真实代码仓库里理解问题、修改代码、运行测试、控制 diff，并避免破坏用户工作区。

典型流程：

1. 准备 repo base commit。
2. 注入 issue、失败测试或需求说明。
3. 启动隔离容器或临时工作区。
4. 给 Agent 文件读写、搜索、Shell、Git、测试工具。
5. 限制网络、密钥和危险命令。
6. 运行 Agent 生成 patch。
7. 应用 patch 并运行测试。
8. 检查 diff 范围、测试结果、安全策略和日志。
9. 输出 score、trace 和失败标签。

常见判分维度：

- 是否解决 issue。
- 相关测试是否通过。
- 是否引入回归。
- diff 是否最小。
- 是否修改无关文件。
- 是否运行必要测试。
- 是否泄露 secret。
- 是否执行危险命令。
- 是否覆盖用户未提交改动。

<h2 id="q-014">面试问题：SWE-bench Harness 给编码 Agent 评测带来哪些启发？</h2>

SWE-bench 的核心启发是：编码 Agent 评测要基于真实仓库、真实 issue、真实测试和可复现环境，而不是只做 LeetCode 式函数生成。

官方 SWE-bench Harness 使用 Docker 构造可复现环境，核心流程包括：

- 为任务准备仓库和依赖环境。
- 应用模型生成的 patch。
- 运行测试命令。
- 根据测试结果判断 issue 是否解决。
- 保存日志和评测结果。

这给企业编码 Harness 的启发：

1. **环境必须可复现**

   使用容器、锁定依赖、固定 base commit。

2. **测试不是唯一指标**

   还要看安全、diff 范围、代码风格、性能和可维护性。

3. **任务要接近真实工作**

   应包含 bug 修复、CI 修复、依赖升级、重构、测试补齐、文档同步。

4. **要防止 benchmark overfitting**

   公开榜单不能替代私有仓库任务集。

5. **日志和失败分类很关键**

   同样失败，可能是定位失败、测试环境失败、工具权限失败或代码生成失败。

面试中可以说：SWE-bench 是编码 Agent Harness 的经典参考，但企业级编码 Harness 还要覆盖私有仓库、权限、dirty worktree、安全策略、review 质量和发布流程。

<h2 id="q-015">面试问题：浏览器 / GUI Agent Harness 应该如何设计？</h2>

浏览器 / GUI Agent Harness 要评估 Agent 是否能理解界面、执行多步操作、处理异常页面，并避免误点击高风险按钮。

核心组件：

- 浏览器自动化环境。
- 页面快照和 DOM 状态。
- 账号与权限隔离。
- 表单、按钮、弹窗、重定向、错误页模拟。
- 页面中的 prompt injection 样本。
- 操作轨迹记录。
- 最终页面状态或后台状态比较。

评测任务：

- 查找信息。
- 填写表单。
- 修改设置。
- 下载文件。
- 生成报表。
- 多页面导航。
- 处理登录和权限不足。
- 遇到支付、删除、提交前请求确认。

指标：

- 任务完成率。
- 错误点击率。
- 高风险动作确认率。
- 注入攻击成功率。
- 平均步骤数。
- 页面恢复能力。
- 超时率。

WebArena、OSWorld 等公开基准说明了网页和 GUI 环境评测的重要性，但企业场景仍需要构造自家 SaaS、后台系统、权限模型和真实流程的私有 Harness。

<h2 id="q-016">面试问题：客服 / 工具调用 Agent Harness 应该如何设计？</h2>

客服 / 工具调用 Agent Harness 的关键不是“回答像不像客服”，而是 Agent 是否在多轮对话中遵守业务政策、正确调用工具，并把系统状态改到期望结果。

tau-bench 是这个方向的重要参考。它把 Agent 放在“用户模拟器 + 领域 API 工具 + 政策规则”的环境中，通过最终数据库状态比较和多次运行稳定性指标评估 Agent。到 2026 年，Sierra Research 还在 τ²-bench / τ³-bench 方向继续扩展，包括更丰富的对话控制、知识检索、语音和多模态客服评测。

典型 Harness 设计：

1. **用户模拟器**

   模拟真实用户追问、补充信息、纠错、反悔、表达不满。

2. **业务工具**

   订单查询、库存查询、退款申请、地址修改、工单创建等。

3. **政策规则**

   退款条件、会员权益、合规话术、禁止承诺。

4. **初始数据库**

   每个任务有独立用户、订单、商品和状态快照。

5. **最终状态判分**

   比较订单、退款、工单、消息状态是否符合预期。

6. **轨迹判分**

   检查是否过度调用工具、是否越权、是否未确认就修改状态。

高频指标：

- task success。
- policy compliance。
- database state match。
- unnecessary tool calls。
- user turns。
- pass@k / consistency。
- escalation accuracy。

面试中可以强调：真实客服 Agent 的 Harness 必须三方闭环评估，即 user-agent-tool，而不是只评估 agent-tool。

<h2 id="q-017">面试问题：多 Agent / A2A Harness 应该如何设计？</h2>

多 Agent Harness 要评估的不只是单个 Agent 能力，还包括任务分配、handoff、协作协议、冲突处理和最终一致性。

需要记录：

- 哪个 Agent 接收任务。
- 是否选择了正确协作者。
- handoff message 是否包含足够上下文。
- remote Agent 返回的 artifact 是否被正确使用。
- 多 Agent 是否重复工作或互相覆盖。
- 失败时是否回退到单 Agent 或人审。
- A2A 调用链是否可追踪。

评测任务：

- 研究 Agent 调用数据分析 Agent。
- 编码 Agent 调用代码审查 Agent。
- 客服 Agent 调用退款审批 Agent。
- 办公 Agent 调用日程、邮件、知识库多个 Agent。

判分维度：

- 最终任务完成率。
- handoff 成功率。
- 上下文丢失率。
- artifact 可用性。
- 冲突解决能力。
- 调用成本。
- 责任边界是否清晰。

多 Agent Harness 最容易漏掉的是“协作成本”。如果多个 Agent 带来大量重复搜索、互相等待和状态不一致，即使最终成功，也可能不适合生产。

---

<h1 id="q-018">5. Agent Harness 如何接入 CI/CD 和发布门禁？</h1>

Agent 发布对象不仅是代码，还包括模型、prompt、tool schema、MCP Server、memory policy、RAG index、guardrail、workflow 和权限策略。Harness 应该成为这些变更的发布门禁。

典型流程：

1. 开发者修改 prompt、工具或策略。
2. 触发 smoke eval。
3. 关键场景通过后运行 regression eval。
4. 安全集和红队样本检查。
5. 与基线版本对比。
6. 指标达标后允许灰度。
7. 灰度阶段持续收集 trace。
8. 线上失败回流到 Harness。

门禁规则示例：

- smoke set 通过率必须 100%。
- regression set 通过率不能下降超过 1%。
- safety set 零高危越权。
- 平均成本不能上升超过 20%。
- P95 延迟不能超过阈值。
- 新版本不得增加敏感信息泄露风险。

<h2 id="q-019">面试问题：Smoke Eval、Regression Eval、Safety Eval、Canary Eval 如何分层？</h2>

| Eval 层级 | 目的 | 运行频率 |
| --- | --- | --- |
| Smoke Eval | 快速发现明显不可用 | 每次提交或 prompt 变更 |
| Regression Eval | 防止历史问题复发 | PR、发布前 |
| Safety Eval | 检查越权、泄密、注入、高风险动作 | 发布前、策略变更前 |
| Robustness Eval | 检查异常工具、网络错误、用户反悔 | 定期或重大变更前 |
| Canary Eval | 小流量真实环境验证 | 灰度阶段 |
| Shadow Eval | 线上旁路评估，不产生副作用 | 生产观测阶段 |

建议分层：

- **小而快**：smoke set，几十个高价值任务。
- **中等规模**：regression set，覆盖历史失败和核心业务。
- **高风险专项**：safety / red-team set。
- **真实分布**：从线上 trace 抽样构造 canary / shadow set。

<h2 id="q-020">面试问题：如何用 Harness 做失败样本回流和持续优化？</h2>

失败样本回流流程：

1. 从线上 trace、用户反馈、人工接管、事故复盘中收集失败。
2. 给失败打标签：理解失败、上下文失败、工具失败、权限失败、策略失败、环境失败。
3. 清洗敏感信息。
4. 固化初始环境和期望状态。
5. 写成可复现 Task Spec。
6. 加入 regression / safety / business set。
7. 修复 prompt、工具、上下文、策略或产品流程。
8. 重新运行 Harness 验证。

注意：

- 不是所有失败都应该靠改 prompt。
- 工具错误应修工具 schema 或返回结构。
- 权限错误应修 policy。
- 上下文错误应修 retrieval / memory / compaction。
- 用户目标模糊应修产品交互和澄清流程。

成熟团队会把 Harness 当作 Agent 的“质量飞轮”：线上失败越多，评测集越强，发布越稳。

<h2 id="q-021">面试问题：Harness 如何服务线上 Shadow Mode 和灰度发布？</h2>

**Shadow Mode** 指新 Agent 旁路运行，不对真实用户产生副作用。它可以用真实流量评估新版本，但不会真正执行写操作。

典型设计：

- 线上请求同时发送给旧版本和新版本。
- 新版本工具调用进入 mock 或 dry-run。
- 比较两者输出、工具计划、成本和安全风险。
- 高风险动作只记录，不执行。
- 抽样人工复核差异。

**灰度发布** 则允许新版本服务一小部分真实用户或低风险任务。

Harness 在灰度中负责：

- 选择灰度任务。
- 记录版本组合。
- 检查 guardrail 命中。
- 对异常 trace 自动标注。
- 触发自动回滚。
- 将灰度失败回流到评测集。

面试中可以说：Shadow Mode 解决“能不能观察真实分布”，Canary 解决“能不能小规模承担真实责任”。

---

<h1 id="q-022">6. 2026 年 Agent Harness Engineering 有哪些主流趋势？</h1>

截至 2026 年，Agent Harness Engineering 的趋势是从“离线脚本评测”走向“模型原生、沙箱原生、trace 原生、平台原生”的质量基础设施。

主流趋势：

1. **Model-native Harness**

   Harness 不只是外部脚本，而是和 Agent SDK、模型工具调用、沙箱、trace 深度集成。

2. **Sandbox-first**

   编码、浏览器、文件和命令执行都在受控沙箱中运行。

3. **Trace Grading**

   对完整轨迹评分，而不是只看最终答案。

4. **State-based Evaluation**

   通过数据库、文件、页面或 artifact 状态判分。

5. **Synthetic User Simulation**

   用用户模拟器评估多轮交互和规则遵循。

6. **Continuous Eval**

   每次 prompt、工具、模型、策略变化都触发评测。

7. **Production Feedback Loop**

   线上 trace 自动转成新 eval case。

8. **Multi-Agent Harness**

   评估 handoff、A2A、协作成本和跨 Agent 状态一致性。

OpenAI 在 2026 年 Agents SDK 演进中也强调了面向文件、命令、代码编辑、长任务和受控沙箱的 model-native harness，这说明 Harness 正在从“评测脚本”升级为 Agent 运行时基础能力。

<h2 id="q-023">面试问题：Model-native Harness 和传统脚本 Harness 有什么区别？</h2>

| 维度 | 传统脚本 Harness | Model-native Harness |
| --- | --- | --- |
| 运行方式 | 外部脚本调用模型 API | 与 Agent SDK / Runtime 深度集成 |
| 工具支持 | 手动 mock 或简单函数 | 原生工具调用、文件、命令、沙箱 |
| Trace | 需要自行记录 | 运行时原生 trace |
| 安全 | 依赖脚本隔离 | 原生权限、sandbox、approval |
| 任务类型 | 偏离线输入输出 | 支持长任务、文件编辑、多工具、多轮 |
| 上线闭环 | 需要额外接入 CI/CD | 更容易接入 eval、trace、deploy gate |

传统脚本 Harness 仍然有价值，尤其适合轻量评测和自定义任务。但复杂 Agent 更需要 Runtime 级 Harness，因为它要评估真实工具执行、上下文组装、权限审批和环境状态变化。

<h2 id="q-024">面试问题：如何回答“请设计一个企业级 Agent Harness 平台”？</h2>

可以按八层架构回答：

1. **Dataset 层**

   管理任务集、风险标签、数据版本、脱敏样本、历史失败样本。

2. **Environment 层**

   提供 Docker / VM / 浏览器 / 数据库 / API mock / 用户模拟器 / 多 Agent 网络。

3. **Runner 层**

   批量运行 Agent，固定版本组合，支持并发、重试、timeout、取消和清理。

4. **Trace 层**

   采集模型调用、工具调用、状态变化、权限审批、成本和 artifact。

5. **Grading 层**

   支持规则判分、测试判分、状态比较、LLM-as-Judge、人审和组合评分。

6. **Report 层**

   展示通过率、失败分类、版本对比、成本、延迟、风险和趋势。

7. **Gate 层**

   接入 CI/CD、Prompt 发布、Tool 发布、模型切换、策略变更和灰度门禁。

8. **Feedback 层**

   将线上失败、人工反馈、事故复盘自动转成新 eval case。

关键设计原则：

- 任务和环境必须版本化。
- 能确定性判分就不要依赖 LLM-as-Judge。
- 每次运行必须有 trace 和 artifact。
- 安全集必须独立于普通功能集。
- 评测要覆盖多次运行稳定性，而不只看 pass@1。
- Harness 不能拥有生产高危权限，只能 dry-run 或在沙箱执行。
- 报告必须能告诉团队“该修模型、工具、上下文、策略还是产品流程”。

**高分总结：**

企业级 Agent Harness 平台的核心不是做排行榜，而是做持续质量门禁。它把 Agent 的每一次变更都变成可复现、可比较、可审计的实验。

---

## 高频速记

1. Harness Engineering 是 Agent 的可复现运行、评估、回放和发布门禁工程。
2. Benchmark 是考卷，Eval 是考试配置，Harness 是考场和阅卷系统，AgentOps 是上线运营体系。
3. Agent Harness 必须评估 outcome、trajectory、state、safety、cost 和 robustness。
4. 高质量任务样本应包含输入、初始环境、工具权限、期望状态、风险标签和判分规则。
5. Environment Simulator / Sandbox 让 Agent 能执行真实动作，但不伤害真实系统。
6. Runner 要固定模型、prompt、工具、策略、环境、随机种子和预算。
7. Trace 是 debug、grading、audit 和失败样本回流的事实依据。
8. Grading 应优先使用确定性规则、测试和状态比较，LLM-as-Judge 负责开放式语义补充。
9. SWE-bench Harness 启发编码 Agent 使用真实仓库、Docker 环境、patch 和测试评估。
10. tau-bench 启发客服 / 工具 Agent 使用用户模拟器、政策规则、API 工具和数据库状态比较。
11. 浏览器 / GUI Harness 要重点检查错误点击、高风险确认、网页注入和最终页面状态。
12. 多 Agent Harness 要评估 handoff、上下文传递、artifact 使用、协作成本和调用链追踪。
13. CI/CD 中应分层运行 smoke、regression、safety、robustness、canary 和 shadow eval。
14. Shadow Mode 旁路评估真实流量但不产生副作用，Canary 小流量承担真实责任。
15. 2026 年趋势是 model-native harness、sandbox-first、trace grading、state-based eval 和 continuous eval。

## 参考资料

- OpenAI, [**Agent evals**](https://platform.openai.com/docs/guides/agent-evals).
- OpenAI, [**Trace grading**](https://platform.openai.com/docs/guides/trace-grading).
- OpenAI, [**The next evolution of the Agents SDK**](https://openai.com/index/the-next-evolution-of-the-agents-sdk), 2026.
- SWE-bench, [**Evaluation Harness Reference**](https://www.swebench.com/SWE-bench/reference/harness/).
- Jimenez et al., [**SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**](https://www.swebench.com/), 2024.
- Zhou et al., [**WebArena: A Realistic Web Environment for Building Autonomous Agents**](https://webarena.dev/), 2024.
- Xie et al., [**OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments**](https://os-world.github.io/), 2024.
- Mialon et al., [**GAIA: A Benchmark for General AI Assistants**](https://huggingface.co/gaia-benchmark), 2023.
- Yao et al., [**tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains**](https://www.tau-bench.com/), 2024.
- Sierra Research, [**tau2-bench / tau3-bench GitHub Repository**](https://github.com/sierra-research/tau2-bench).
