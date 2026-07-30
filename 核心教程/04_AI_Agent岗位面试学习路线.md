# 我要面试 AI Agent 岗，先看哪些？

> 路线定位：面向 AI Agent 算法、应用开发、平台工程、AgentOps、编码 Agent 等岗位。建议 4-6 周完成，每周至少进行一次系统设计口述。

[返回核心教程总览](README.md) | [返回项目主页](../README.md)

## 面试目标：从“会用框架”升级到“能解释 Agent 系统”

Agent 岗的核心不是背 LangGraph、Dify 或某个 SDK 的 API，而是理解一个模型如何在约束环境中**感知状态、规划任务、选择工具、执行动作、读取反馈、更新上下文并接受评测**。框架会变化，这条闭环和其中的工程权衡具有更强的跨周期价值。

面试准备建议分成六层：

1. Agent 本质与运行循环。
2. Tool Use、Function Calling 与协议。
3. 规划、工作流与多 Agent 编排。
4. 上下文、Memory 与状态持久化。
5. 安全、评测、Trace 与 AgentOps。
6. 企业平台、生产落地与系统设计。

## 第 1 层：Agent 本质与基本循环

### 必须掌握

- [AIGC 时代 AI Agent 的本质](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-001>)
- [Agent、Chatbot、Workflow 与 Copilot 的区别](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-002>)
- [Agent 核心技术栈如何拆解](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-003>)
- [Agent 基本运行循环](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-005>)
- [ReAct、Plan-and-Execute、Reflection 与 ToT 对比](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-006>)
- [为什么 Agent 需要 Human-in-the-loop](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-010>)

### 高频追问

- 什么任务不应该使用 Agent？
- Agent 的“自主性”应该由谁控制？
- 为什么一个复杂工作流不等于 Agent？
- 模型能力增强后，哪些编排仍必须保留在代码层？

### 通过标准

能在 3 分钟内画出 Observe -> Reason/Plan -> Act -> Feedback -> State Update 循环，并说明终止条件、错误路径与人工接管点。

## 第 2 层：Function Calling、工具系统与协议

### 必须掌握

- [Function Calling、Tool Use 与 Structured Output 的区别](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-007>)
- [工具 Schema 的高频问题](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-008>)
- [Agent 如何选择工具并处理失败](<../AI%20Agent基础/01_AIGC时代AI%20Agent基础高频考点.md#q-009>)
- [Function Calling 和传统 API 调用的区别](<../AI%20Agent基础/04_MCP与A2A协议高频考点.md#q-006>)
- [MCP Host、Client、Server 的职责](<../AI%20Agent基础/04_MCP与A2A协议高频考点.md#q-011>)
- [MCP Tools、Resources、Prompts 的区别](<../AI%20Agent基础/04_MCP与A2A协议高频考点.md#q-012>)
- [MCP 与 A2A 分别解决什么问题](<../AI%20Agent基础/04_MCP与A2A协议高频考点.md#q-002>)
- [什么时候不应该使用 MCP 或 A2A](<../AI%20Agent基础/04_MCP与A2A协议高频考点.md#q-027>)

### 高频追问

- 如何设计工具粒度，避免“万能工具”和“碎片工具”？
- 工具调用失败后，重试、换工具、降级和转人工如何选择？
- 如何保证写操作幂等？
- MCP 解决互操作后，为什么认证、权限和治理仍需要平台层？

### 通过标准

能设计一个高质量工具 Schema，包含明确语义、参数约束、错误结构、超时、幂等键、权限要求和副作用说明。

## 第 3 层：设计模式、规划与工作流编排

### 必须掌握

- [Agent Pattern、Prompt Pattern 与 Workflow Pattern](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-002>)
- [Agent 和 Workflow 如何取舍](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-003>)
- [为什么 Agent 项目会失败在过度自主](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-004>)
- [ReAct 的完整运行循环](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-006>)
- [Plan-and-Solve 与 Plan-and-Execute 的区别](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-011>)
- [Reflection/Reflexion 如何提升可靠性](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-013>)
- [多 Agent 的 Supervisor、Router、Swarm 与 Debate](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-021>)
- [为什么多 Agent 不一定优于单 Agent](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-023>)
- [如何设计可恢复的 Agent 工作流](<../AI%20Agent基础/03_Agent设计模式与工作流编排高频考点.md#q-024>)

### 面试判断框架

面对业务场景时，按以下顺序选择：

1. 固定规则能否完成？能，则使用普通程序。
2. 路径固定、内容不确定？使用 LLM Workflow。
3. 路径需要动态决策、工具受控？使用单 Agent。
4. 是否存在真正独立的角色、上下文或权限边界？有，才考虑多 Agent。

### 通过标准

能把一个“全自主多 Agent”方案主动收敛成更可靠的工作流，并解释收敛后为什么效果、成本和可测试性更好。

## 第 4 层：Context Engineering、Memory 与状态

### 必须掌握

- [Prompt Engineering 与 Context Engineering 的区别](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-002>)
- [长上下文为什么不能替代记忆系统](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-004>)
- [Memory 的主要类型](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-006>)
- [Memory 与 RAG 的本质区别](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-008>)
- [如何判断信息是否写入长期记忆](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-011>)
- [记忆冲突、过期和遗忘如何处理](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-014>)
- [Context Engine 的 assemble/compact/retrieve](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-018>)
- [向量库、关系库、图数据库与事件日志如何分工](<../AI%20Agent基础/05_Agent记忆与上下文工程高频考点.md#q-021>)

### 高频追问

- 用户偏好、任务状态、知识文档和工具返回应该存在哪里？
- 摘要压缩丢失关键约束时如何恢复？
- 如何防止错误经验被长期记忆放大？
- 多租户情况下如何做记忆隔离与删除？

### 通过标准

能设计一套 Memory 生命周期：写入判断、存储、检索、更新、冲突处理、遗忘、审计和删除，并说明每一步的质量风险。

## 第 5 层：安全、评测、Trace 与 AgentOps

### 必须掌握

- [Agent 风险边界与 Chatbot 的区别](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-002>)
- [为什么 Agent 评测不能只看最终答案](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-003>)
- [输入、输出、工具、状态四类 Guardrails](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-006>)
- [工具权限、沙箱与人工审批](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-008>)
- [如何构建企业内部 Agent Eval Harness](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-014>)
- [如何通过 Trace 定位失败](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-016>)
- [如何建立失败样本回流](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-019>)
- [如何分阶段上线 Agent](<../AI%20Agent基础/06_Agent安全评测与AgentOps高频考点.md#q-021>)
- [Agent Harness 的完整模块](<../AI%20Agent基础/08_Agent_Harness_Engineering高频考点.md#q-004>)
- [Outcome、Trajectory 与 State Grading](<../AI%20Agent基础/08_Agent_Harness_Engineering高频考点.md#q-010>)

### 评测最小集合

一个完整 Agent 项目至少报告：任务成功率、工具选择正确率、参数正确率、步骤数、重试率、人工接管率、P95 延迟、单任务成本、安全违规率和失败类型分布。

### 通过标准

给你一条失败 trace，你能判断是模型推理、工具、环境、状态、权限还是终止条件导致；给你一个新版本，你能设计回归门禁而不是直接全量发布。

## 第 6 层：企业平台与系统设计

### 必须掌握

- [Agent 平台和单个 Agent 应用的区别](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-002>)
- [Agent Builder 的能力边界](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-006>)
- [Tool、MCP 与 Agent Registry 如何分工](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-007>)
- [会话、任务、记忆与知识库为什么分开管理](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-009>)
- [RBAC、ABAC 与租户隔离](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-011>)
- [如何评估 Agent 场景是否值得做](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-018>)
- [平台如何支持长任务和后台任务](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-024>)
- [如何设计企业级 Agent 平台](<../AI%20Agent基础/07_企业级Agent平台与产品落地高频考点.md#q-028>)

### 系统设计答题顺序

1. 澄清用户、场景、成功指标与风险等级。
2. 确定 Workflow/Agent 边界与人工接管点。
3. 画出接入层、编排层、模型层、工具层、状态层和数据层。
4. 补充权限、隔离、审计、评测与可观测性。
5. 说明成本、延迟、并发、降级、灰度和回滚。
6. 用失败路径验证架构，而不是只描述正常路径。

## 加分专题：编码 Agent 与长期运行 Agent

- [编码 Agent 的典型架构](<../AI%20Agent基础/02_编码Agent与AgentOS工程高频考点.md#q-003>)
- [编码 Agent 如何理解真实代码仓库](<../AI%20Agent基础/02_编码Agent与AgentOS工程高频考点.md#q-010>)
- [如何设计企业可用的编码 Agent](<../AI%20Agent基础/02_编码Agent与AgentOS工程高频考点.md#q-029>)
- [AgentOS 的分层方式](<../AI%20Agent基础/02_编码Agent与AgentOS工程高频考点.md#q-030>)
- [经验如何沉淀为 Memory、Skill 和 Policy](<../AI%20Agent基础/09_自进化Agent与多平台运行时高频考点.md#q-003>)
- [如何设计自进化、多平台 Agent](<../AI%20Agent基础/09_自进化Agent与多平台运行时高频考点.md#q-024>)

## 4 周最小冲刺安排

| 周次 | 学习重点 | 必须产出 |
|---|---|---|
| 第 1 周 | Agent 本质、Tool Use、MCP | Agent 循环图、工具 Schema、20 道口述题 |
| 第 2 周 | Workflow、规划、Memory | 状态机图、Memory 设计、可恢复任务 Demo |
| 第 3 周 | Eval、Trace、安全、AgentOps | 评测集、Trace 示例、失败分类报告 |
| 第 4 周 | 企业平台与系统设计 | 完整架构图、项目复盘、两轮模拟面试 |

## 最终验收清单

- [ ] 我能区分 Chatbot、Workflow、单 Agent 与多 Agent。
- [ ] 我能设计工具 Schema、权限、错误处理与幂等机制。
- [ ] 我能解释 MCP、A2A、Memory、RAG 和 Context Engineering 的边界。
- [ ] 我能设计结果、轨迹和状态三层评测，并通过 Trace 定位失败。
- [ ] 我能从业务价值、安全等级和工程成本判断一个 Agent 场景是否值得做。
- [ ] 我能在 30 分钟内完成企业级 Agent 系统设计，并经受追问。

Agent 面试的最终分水岭，不是你调用过多少框架，而是你是否能把一个概率性执行系统放进真实约束中，并让它可控、可测、可追踪、可演进。
