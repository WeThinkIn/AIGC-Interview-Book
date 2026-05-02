# 目录

## 第一章 AI Agent 总览

[1. AIGC 时代 AI Agent 的本质是什么？](#q-001)
  - [面试问题：Agent、Chatbot、Workflow、Copilot 有什么区别？](#q-002)
  - [面试问题：AI Agent 的核心技术栈如何拆解？](#q-003)
  - [面试问题：为什么 2025-2026 年 Agent 从 Demo 走向工程系统？](#q-004)

## 第二章 Agent 循环、规划与工具使用

[2. Agent 的基本运行循环是什么？](#q-005)
  - [面试问题：ReAct、Plan-and-Execute、Reflection、Tree-of-Thought 如何对比？](#q-006)
  - [面试问题：Function Calling、Tool Use、Structured Output 有什么区别？](#q-007)
  - [面试问题：工具 Schema 设计有哪些高频坑？](#q-008)
  - [面试问题：Agent 如何选择工具、调用工具并处理工具失败？](#q-009)
  - [面试问题：Agent 为什么需要 Human-in-the-loop？](#q-010)

## 第三章 MCP、A2A 与 Agent 标准化

[3. MCP 为什么成为 Agent 工具生态的重要协议？](#q-011)
  - [面试问题：MCP 和 Function Calling 的区别是什么？](#q-012)
  - [面试问题：MCP 的 stdio、Streamable HTTP、SSE 传输如何选择？](#q-013)
  - [面试问题：A2A 和 MCP 分别解决什么问题？](#q-014)
  - [面试问题：AGENTS.md、Skills、Memory 文件为什么重要？](#q-015)

## 第四章 主流 Agent 框架与产品生态

[4. 当前主流 Agent 框架如何分类？](#q-016)
  - [面试问题：OpenAI Agents SDK 的核心能力有哪些？](#q-017)
  - [面试问题：LangGraph 为什么适合复杂 Agent 工作流？](#q-018)
  - [面试问题：CrewAI、AutoGen、Google ADK、LlamaIndex Agent 如何对比？](#q-019)
  - [面试问题：Claude Code、Codex、Cursor、Devin、Jules 代表什么趋势？](#q-020)

## 第五章 记忆、上下文工程与知识增强

[5. Agent 为什么需要上下文工程？](#q-021)
  - [面试问题：短期记忆、长期记忆、任务记忆、工具记忆如何区分？](#q-022)
  - [面试问题：Memory 和 RAG 的区别如何快速回答？](#q-023)
  - [面试问题：上下文窗口溢出时如何处理？](#q-024)
  - [面试问题：Agent 如何做自我总结、状态恢复和跨会话连续性？](#q-025)

## 第六章 安全、评估与工程落地

[6. Agent 系统为什么必须重视权限和安全？](#q-026)
  - [面试问题：Agent Guardrails 的边界如何快速概括？](#q-027)
  - [面试问题：Agent 可观测性和 tracing 应该记录什么？](#q-028)
  - [面试问题：Agent 常用评测基准有哪些？](#q-029)
  - [面试问题：如何设计一个企业级 AI Agent 系统？](#q-030)

---

<h1 id="q-001">1. AIGC 时代 AI Agent 的本质是什么？</h1>

AI Agent 是以大模型为核心，能够围绕目标进行感知、规划、调用工具、执行动作、观察反馈、维护状态并持续迭代的智能系统。

一个简化定义：

$$
\text{Agent} = \text{LLM} + \text{Tools} + \text{Memory} + \text{Planning} + \text{Control Loop} + \text{Environment}
$$

相比普通大模型问答，Agent 的关键不在“回答一句话”，而在“完成一个任务”。它需要：

- 理解用户目标。
- 拆解任务步骤。
- 选择合适工具。
- 调用外部 API、数据库、浏览器、文件系统或代码环境。
- 根据工具结果修正计划。
- 在多轮交互中保持状态。
- 在危险操作前请求确认。
- 输出可验证结果。

**面试金句：**

大模型是 Agent 的大脑，工具是手脚，记忆是上下文连续性，规划是任务分解，权限和评估是工程落地的刹车系统。

<h2 id="q-002">面试问题：Agent、Chatbot、Workflow、Copilot 有什么区别？</h2>

| 概念 | 核心特点 | 适合场景 | 风险 |
| --- | --- | --- | --- |
| Chatbot | 多轮问答，主要生成文本 | FAQ、客服、知识问答 | 不擅长真实执行 |
| Workflow | 预定义流程，步骤固定 | 审批、报表、数据处理 | 灵活性有限 |
| Copilot | 人机协作，用户主导 | 编码、写作、办公辅助 | 依赖用户判断 |
| Agent | 目标驱动，自主规划和工具执行 | 研究、编码、运营、自动化任务 | 不可控风险更高 |

判断标准：

- 如果流程固定，优先 Workflow。
- 如果需要用户持续监督，适合 Copilot。
- 如果需要系统自主拆解、执行和迭代，才需要 Agent。
- 如果只是问答，不要过度设计成 Agent。

很多产品失败不是因为 Agent 不够智能，而是把确定性工作流错误地做成了不可控 Agent。

<h2 id="q-003">面试问题：AI Agent 的核心技术栈如何拆解？</h2>

AI Agent 可以拆成九层：

1. **模型层**

   负责推理、工具选择、代码理解、规划和自然语言交互。常用闭源/开源大模型、长上下文模型、代码模型、多模态模型。

2. **提示词与指令层**

   包括 system prompt、developer prompt、任务 prompt、工具说明、策略约束、输出格式和角色边界。

3. **工具层**

   包括 function calling、MCP、浏览器、Shell、代码执行器、数据库、文件系统、搜索、业务 API。

4. **规划层**

   包括 ReAct、Plan-and-Execute、任务树、Todo List、子 Agent 分工、动态重规划。

5. **记忆层**

   包括会话历史、任务状态、长期偏好、项目知识、向量库、图谱记忆和摘要。

6. **上下文工程层**

   决定哪些信息进入当前上下文，如何压缩、检索、排序、裁剪和注入。

7. **权限与安全层**

   包括工具审批、沙箱、敏感数据检测、越权防护、审计日志、策略执行。

8. **执行与编排层**

   包括状态机、工作流引擎、任务队列、重试、超时、并发、回滚、持久化。

9. **评估与观测层**

   包括 tracing、日志、工具调用记录、成本、延迟、成功率、人工验收和离线评测。

<h2 id="q-004">面试问题：为什么 2025-2026 年 Agent 从 Demo 走向工程系统？</h2>

原因主要有六个：

1. **模型工具调用能力增强**

   主流模型能更稳定地产生 JSON、调用函数、理解工具错误并修复计划。

2. **协议标准化**

   MCP 标准化了“模型应用如何接入工具和上下文”；A2A 标准化了“Agent 服务之间如何通信”。

3. **框架成熟**

   OpenAI Agents SDK、LangGraph、CrewAI、AutoGen、Google ADK 等框架提供了 handoff、guardrails、tracing、memory、durable execution 等工程能力。

4. **编码 Agent 证明价值**

   Claude Code、Codex、Cursor、Devin、Jules 等工具让 Agent 在真实代码仓库中执行读写、测试、调试和提交。

5. **评测体系更清晰**

   SWE-bench、WebArena、OSWorld、GAIA、τ-bench 等让 Agent 不再只靠 demo 评估。

6. **企业需求明确**

   企业更关心自动化办公、数据分析、客服、研发提效、运营执行和知识工作流，这些都适合工具型 Agent。

---

<h1 id="q-005">2. Agent 的基本运行循环是什么？</h1>

经典 Agent 循环可以表示为：

$$
\text{Observe} \rightarrow \text{Think/Plan} \rightarrow \text{Act} \rightarrow \text{Observe} \rightarrow \cdots
$$

工程上通常是：

1. 用户提出目标。
2. 系统构造上下文：指令、历史、记忆、工具列表、环境状态。
3. 模型生成下一步：回答、工具调用、计划更新或请求澄清。
4. 工具执行器执行动作。
5. 系统收集工具结果。
6. 模型根据结果继续推理。
7. 达到停止条件后输出最终结果。

停止条件包括：

- 任务完成。
- 达到最大轮数。
- 达到预算上限。
- 工具失败不可恢复。
- 需要用户授权或补充信息。
- 触发安全策略。

<h2 id="q-006">面试问题：ReAct、Plan-and-Execute、Reflection、Tree-of-Thought 如何对比？</h2>

本节只做面试总览，帮助快速识别不同模式的适用边界；ReAct、Plan-and-Solve、ReWOO、Reflection、ToT、GoT、LATS 和多 Agent 编排的完整推导，见 [03_Agent设计模式与工作流编排高频考点.md](03_Agent设计模式与工作流编排高频考点.md)。

| 模式 | 核心思想 | 优点 | 局限 |
| --- | --- | --- | --- |
| ReAct | Reasoning 和 Acting 交替 | 简单、适合工具使用 | 容易局部贪心 |
| Plan-and-Execute | 先制定计划，再分步执行 | 结构清晰，适合长任务 | 初始计划可能过时 |
| Reflection | 执行后自评和修正 | 能从错误中恢复 | 增加成本和延迟 |
| Tree-of-Thought | 多路径搜索和比较 | 适合复杂推理 | 成本高，工程复杂 |
| Multi-Agent | 多角色协作 | 适合专业分工 | 协调成本和一致性问题 |

实际系统常常混合使用：

- 简单任务：ReAct。
- 长任务：Plan-and-Execute + 动态重规划。
- 高风险任务：Reflection + 人类确认。
- 复杂研究或编码：主 Agent + 子 Agent + 工具执行。

<h2 id="q-007">面试问题：Function Calling、Tool Use、Structured Output 有什么区别？</h2>

**Structured Output**：要求模型输出符合某个 JSON Schema 或结构化格式，不一定执行外部动作。

**Function Calling**：模型选择一个函数并生成参数，由宿主程序执行函数。

**Tool Use**：更广义，工具可以是函数、浏览器、代码解释器、Shell、MCP 工具、文件系统、GUI 操作或远程服务。

对比：

| 能力 | 是否执行外部动作 | 典型用途 |
| --- | --- | --- |
| Structured Output | 不一定 | 抽取、分类、计划、表单 |
| Function Calling | 是 | 调业务 API、查数据库、发邮件 |
| Tool Use | 是 | 搜索、浏览器、Shell、代码、MCP、GUI |

面试中可以说：Function Calling 是 Tool Use 的一种标准化接口，Structured Output 是让模型输出可被程序可靠解析的基础能力。

<h2 id="q-008">面试问题：工具 Schema 设计有哪些高频坑？</h2>

常见问题：

1. **工具描述太模糊**

   模型不知道何时调用、何时不调用。

2. **参数过多或层级太深**

   增加模型填参错误概率。

3. **枚举值不清晰**

   模型容易创造不存在的取值。

4. **缺少前置条件**

   例如删除文件前必须确认、付款前必须校验用户授权。

5. **工具粒度不合适**

   太细导致调用链很长；太粗导致不可控。

6. **错误信息不可恢复**

   工具失败后只返回 “failed”，模型不知道如何修复。

好的工具设计应该满足：

- 名称动作明确。
- 描述包含使用边界。
- 参数少而清晰。
- 返回结构可解析。
- 错误信息可行动。
- 高风险工具有权限策略。

<h2 id="q-009">面试问题：Agent 如何选择工具、调用工具并处理工具失败？</h2>

推荐流程：

1. **工具发现**

   将当前可用工具、能力范围、参数 Schema 注入上下文，或通过 MCP/tool search 动态发现。

2. **工具选择**

   模型根据任务意图选择工具；对于高风险工具，先解释意图并等待授权。

3. **参数生成**

   使用 JSON Schema 约束参数；必要时让模型先生成计划，再生成工具参数。

4. **执行与观测**

   工具执行器返回结构化结果、错误码、stderr/stdout、资源链接或状态变更。

5. **失败恢复**

   根据错误类型重试、换工具、缩小范围、请求用户输入或终止。

6. **结果验证**

   检查是否满足目标，例如测试是否通过、文件是否存在、数据库状态是否符合预期。

工具失败分类：

- 参数错误：让模型修正参数。
- 权限错误：请求用户授权或降级。
- 环境错误：提示依赖缺失。
- 网络错误：重试或换源。
- 语义错误：重新规划。
- 安全错误：停止执行。

<h2 id="q-010">面试问题：Agent 为什么需要 Human-in-the-loop？</h2>

Agent 的不确定性来自模型推理、工具执行、环境变化和用户目标模糊。Human-in-the-loop 是把人类判断放在关键边界上。

典型场景：

- 删除、覆盖、发布、支付、发邮件等不可逆操作。
- 访问敏感数据。
- 低置信度或多方案选择。
- 需求不明确。
- 安全策略触发。
- 需要业务负责人审批。

交互方式：

- 让用户确认计划。
- 在工具执行前弹出权限审批。
- 生成 diff 后等待确认。
- 给出多个方案供选择。
- 失败时请求补充信息。

面试要点：企业级 Agent 不追求完全无人值守，而是把自主执行限制在可控边界内。

---

<h1 id="q-011">3. MCP 为什么成为 Agent 工具生态的重要协议？</h1>

本章只解释 MCP / A2A 在 Agent 技术地图中的位置；MCP Host/Client/Server、Tools/Resources/Prompts、Roots、Sampling、Elicitation、stdio、Streamable HTTP、A2A Agent Card 和协议安全细节，统一见 [04_MCP与A2A协议高频考点.md](04_MCP与A2A协议高频考点.md)。

MCP，全称 Model Context Protocol，是一种开放协议，用来标准化模型应用如何接入外部工具、数据源和上下文。

没有 MCP 时，每个 Agent 应用都要为 GitHub、数据库、浏览器、文件系统、企业知识库写一套私有连接器。MCP 的价值是把这些连接器变成标准服务器：

$$
\text{Agent Client} \leftrightarrow \text{MCP Server} \leftrightarrow \text{Tool/Data Source}
$$

MCP 让工具生态从“每个应用单独集成”走向“工具服务器可复用”。

<h2 id="q-012">面试问题：MCP 和 Function Calling 的区别是什么？</h2>

| 维度 | Function Calling | MCP |
| --- | --- | --- |
| 抽象层级 | 模型调用宿主程序定义的函数 | Agent 客户端连接标准化工具服务器 |
| 工具来源 | 应用内本地定义 | 本地或远程 MCP Server |
| 协议 | 由模型 API/SDK 提供 | JSON-RPC + 标准能力协商 |
| 复用性 | 通常绑定某个应用 | 跨应用复用 |
| 能力范围 | 函数调用 | tools、resources、prompts、sampling 等 |
| 适合场景 | 简单业务 API | 工具生态、企业连接器、跨应用工具共享 |

可以这样回答：

Function Calling 是模型调用函数的能力；MCP 是把工具和上下文服务标准化暴露给模型应用的协议。

<h2 id="q-013">面试问题：MCP 的 stdio、Streamable HTTP、SSE 传输如何选择？</h2>

| 传输 | 特点 | 适合场景 |
| --- | --- | --- |
| stdio | 客户端启动本地子进程，通过 stdin/stdout 通信 | 本地工具、CLI、开发环境、低延迟 |
| Streamable HTTP | 独立 HTTP 服务，可支持多客户端、远程部署和流式响应 | 企业工具服务、云端 MCP、多用户 |
| SSE | 旧版 HTTP+SSE 路线，逐步被 Streamable HTTP 替代 | 兼容旧实现 |

安全注意：

- 本地 Streamable HTTP 应绑定 localhost，避免暴露到公网。
- 远程 MCP 必须有认证、授权和审计。
- 工具返回内容要防 prompt injection。
- 高风险工具要有 approval policy。

<h2 id="q-014">面试问题：A2A 和 MCP 分别解决什么问题？</h2>

MCP 解决“Agent 如何使用工具和上下文”的问题。

A2A 解决“Agent 如何和其他 Agent 服务通信协作”的问题。

| 维度 | MCP | A2A |
| --- | --- | --- |
| 连接对象 | 工具、数据源、上下文服务器 | 独立 Agent 服务 |
| 关系 | Agent-to-Tool | Agent-to-Agent |
| 核心能力 | 工具调用、资源访问、提示模板 | 任务委托、状态通信、远程能力调用 |
| 典型场景 | 连接 GitHub、数据库、文件系统 | 调用金融分析 Agent、客服 Agent、代码审查 Agent |

本地子 Agent 通常在同一进程内通信，延迟低，适合模块化分工；远程 Agent 更适合跨团队、跨服务、跨组织协作。

<h2 id="q-015">面试问题：AGENTS.md、Skills、Memory 文件为什么重要？</h2>

Agent 不只需要工具，还需要知道“这个项目怎么工作”。AGENTS.md、Skills、Memory 文件解决的是项目级上下文和可复用知识的问题。

**AGENTS.md**：面向编码 Agent 的项目说明文件，包含构建命令、测试命令、代码规范、目录结构、提交要求等。

**Skills**：可按需加载的能力包，通常包含 `SKILL.md`、脚本、模板、参考资料，让 Agent 在特定任务中获得专业流程。

**Memory 文件**：保存长期偏好、项目约定、团队知识、历史决策，帮助跨会话复用经验。

三者对比：

| 类型 | 解决问题 | 生命周期 |
| --- | --- | --- |
| AGENTS.md | 项目级通用指令 | 随代码仓库长期存在 |
| Skill | 任务级专业能力 | 按需加载和复用 |
| Memory | 个体/团队历史经验 | 持续积累和更新 |

---

<h1 id="q-016">4. 当前主流 Agent 框架如何分类？</h1>

可以按工程定位分为五类：

1. **通用 Agent SDK**

   代表：OpenAI Agents SDK。重点是 agent、tool、handoff、guardrails、tracing、MCP。

2. **图式工作流/状态机框架**

   代表：LangGraph。重点是状态图、持久化执行、人类介入、可恢复任务。

3. **多 Agent 协作框架**

   代表：CrewAI、AutoGen、Google ADK。重点是角色分工、通信、协作、任务编排。

4. **知识型 Agent/RAG 框架**

   代表：LlamaIndex Agent/Workflow。重点是数据连接、检索、索引、知识增强。

5. **编码 Agent 产品**

   代表：Claude Code、Codex、Cursor、Devin、Jules。重点是真实代码仓库读写、测试、提交、审查和任务执行。

<h2 id="q-017">面试问题：OpenAI Agents SDK 的核心能力有哪些？</h2>

OpenAI Agents SDK 面向生产级 Agent 编排，核心能力包括：

- **Agent**：定义模型、指令、工具、输出类型和 handoffs。
- **Tools**：函数工具、托管工具、MCP 工具、代码/浏览器/文件等执行能力。
- **Handoffs**：把任务转交给专门 Agent。
- **Guardrails**：对输入、输出、工具调用做安全和质量检查。
- **Tracing**：记录模型调用、工具调用、handoff、guardrail、自定义事件。
- **Sessions/Memory**：保存多轮会话状态。
- **Structured Output**：约束最终输出格式。

面试重点：

OpenAI Agents SDK 把 Agent 从“提示词 + 函数调用”提升到“可观测、可防护、可编排的工程系统”。

<h2 id="q-018">面试问题：LangGraph 为什么适合复杂 Agent 工作流？</h2>

LangGraph 的核心是把 Agent 工作流表示为有状态图：

- 节点是 LLM、工具、判断逻辑或子流程。
- 边表示控制流。
- State 保存中间状态。
- Checkpointer 支持持久化和恢复。
- Human-in-the-loop 可以插入审批和修改。

适合场景：

- 多步骤业务流程。
- 长时间运行任务。
- 需要失败恢复。
- 需要可视化和可观测。
- 需要人工审批。
- 需要多个 Agent/工具协作。

与普通链式调用相比，LangGraph 更像“Agent 工作流引擎”，而不是简单 prompt pipeline。

<h2 id="q-019">面试问题：CrewAI、AutoGen、Google ADK、LlamaIndex Agent 如何对比？</h2>

| 框架 | 核心定位 | 适合场景 |
| --- | --- | --- |
| CrewAI | Crew + Agent + Task + Process | 角色分工明确的业务流程 |
| AutoGen | 多 Agent 对话和协作 | 研究型多 Agent、代码协作、模拟讨论 |
| Google ADK | Agent 开发工具包，支持 A2A | Google/Gemini 生态、多 Agent 服务化 |
| LlamaIndex Agent | 数据连接和知识增强 | RAG Agent、企业知识库、文档问答 |
| LangGraph | 有状态图和持久执行 | 复杂工作流、长期任务、human-in-loop |
| OpenAI Agents SDK | 通用 Agent SDK | 工具调用、handoff、guardrail、tracing |

选择建议：

- 数据密集型：LlamaIndex。
- 状态复杂：LangGraph。
- 多角色协作：CrewAI/AutoGen。
- 标准 SDK 和工具生态：OpenAI Agents SDK。
- Google 生态和 A2A：Google ADK。

<h2 id="q-020">面试问题：Claude Code、Codex、Cursor、Devin、Jules 代表什么趋势？</h2>

这些工具代表 Agent 在软件工程中的产品化。

共同趋势：

- 从“代码补全”走向“任务执行”。
- 从单文件编辑走向仓库级理解。
- 从聊天窗口走向 IDE/CLI/云端工作区。
- 从建议代码走向运行测试、修 bug、提 PR。
- 从单 Agent 走向子 Agent、技能、hooks、项目说明文件。
- 从一次性对话走向会话恢复、记忆和长期任务。

核心能力：

- 代码搜索和索引。
- 文件读写和 diff。
- Shell/测试/构建执行。
- Git 操作。
- 权限审批。
- 上下文压缩。
- 项目级指令。
- 工具结果摘要。
- 远程会话和 IDE 桥接。

编码 Agent 是 Agent 工程最成熟的落地场景之一，因为软件仓库天然具备可验证反馈：编译、测试、lint、diff、CI。

---

<h1 id="q-021">5. Agent 为什么需要上下文工程？</h1>

本节只给出上下文工程的总览视角；Memory 类型、Memory vs RAG、Context Engine、记忆生命周期、存储架构、隐私治理和 Skills 渐进式上下文，统一见 [05_Agent记忆与上下文工程高频考点.md](05_Agent记忆与上下文工程高频考点.md)。

上下文工程是决定“当前这次模型调用应该看到什么”的技术。

Agent 面临的问题：

- 上下文窗口有限。
- 工具结果可能很长。
- 历史对话会污染当前任务。
- 项目文件太多。
- 记忆可能过期或冲突。
- 不同工具需要不同上下文。

上下文工程的目标：

- 保留任务相关信息。
- 去掉无关噪声。
- 控制 token 成本。
- 提高工具选择准确性。
- 降低幻觉和遗忘。
- 让长任务可持续执行。

<h2 id="q-022">面试问题：短期记忆、长期记忆、任务记忆、工具记忆如何区分？</h2>

| 记忆类型 | 内容 | 用途 |
| --- | --- | --- |
| 短期记忆 | 当前对话、最近工具结果 | 当前轮推理 |
| 长期记忆 | 用户偏好、项目约定、长期事实 | 跨会话连续性 |
| 任务记忆 | 当前任务计划、进度、阻塞点 | 长任务执行 |
| 工具记忆 | 工具可用性、参数经验、失败记录 | 提高工具调用成功率 |
| 团队记忆 | 团队规范、共享知识 | 多人/多 Agent 协作 |

工程上需要解决：

- 何时写入记忆。
- 写入前如何去重和脱敏。
- 何时检索。
- 记忆冲突如何处理。
- 用户如何查看、编辑和删除记忆。

<h2 id="q-023">面试问题：Memory 和 RAG 的区别如何快速回答？</h2>

RAG 主要解决“从外部知识库检索事实”；Memory 主要解决“Agent 在交互和执行中积累状态与经验”。

| 维度 | RAG | Memory |
| --- | --- | --- |
| 数据来源 | 文档、知识库、网页、数据库 | 对话、行为、偏好、任务状态 |
| 更新频率 | 相对低 | 高，随交互变化 |
| 目标 | 补充知识 | 保持连续性和个性化 |
| 检索方式 | Query-driven | Context/State-driven |
| 风险 | 检索不准、文档过期 | 记忆污染、隐私泄露 |

两者可以结合：RAG 提供外部知识，Memory 提供个体和任务上下文。

<h2 id="q-024">面试问题：上下文窗口溢出时如何处理？</h2>

常见策略：

1. **滑动窗口**

   保留最近消息，丢弃早期内容。简单但容易丢失关键决策。

2. **摘要压缩**

   将历史对话、工具结果、任务状态压缩成摘要。

3. **语义检索**

   将历史消息和文件切片向量化，按需检索。

4. **结构化状态**

   把任务进度、TODO、约束、已完成动作保存为结构化对象。

5. **工具结果折叠**

   长 stdout、搜索结果、文件内容只保留摘要和引用。

6. **分层上下文**

   系统指令、任务目标、计划、相关文件、历史摘要按优先级拼接。

7. **子 Agent 隔离**

   把探索性任务交给子 Agent，主 Agent 只接收结论。

<h2 id="q-025">面试问题：Agent 如何做自我总结、状态恢复和跨会话连续性？</h2>

典型方案：

- 每次长任务结束时生成 session summary。
- 将计划、已完成步骤、未完成事项、关键文件、失败尝试保存为结构化状态。
- 下次 resume 时先注入摘要，再按需加载历史细节。
- 对项目约定和用户偏好写入长期记忆。
- 对工具失败和环境信息保存为诊断记录。

好的恢复摘要应包含：

- 用户目标。
- 当前进度。
- 修改过的文件。
- 已运行的命令和结果。
- 未解决问题。
- 下一步建议。
- 重要约束和风险。

---

<h1 id="q-026">6. Agent 系统为什么必须重视权限和安全？</h1>

本节只做风险地图和高频概念索引；Guardrails、Prompt Injection、权限沙箱、Benchmark、Eval Harness、Tracing、灰度发布、事故响应和 AgentOps 的工程细节，统一见 [06_Agent安全评测与AgentOps高频考点.md](06_Agent安全评测与AgentOps高频考点.md)。企业级平台架构与产品落地见 [07_企业级Agent平台与产品落地高频考点.md](07_企业级Agent平台与产品落地高频考点.md)。

Agent 可以调用工具，因此它不仅会“说错话”，还可能“做错事”。安全边界比普通 Chatbot 更重要。

高风险包括：

- 删除或覆盖文件。
- 泄露密钥、隐私、商业数据。
- 执行恶意命令。
- 调用生产 API。
- 发送邮件或消息。
- 下单、转账、审批。
- 被网页或工具返回内容 prompt injection。

因此生产级 Agent 必须有：

- 最小权限。
- 沙箱。
- 工具审批。
- 审计日志。
- 输入输出 guardrails。
- 敏感信息检测。
- 速率和预算限制。
- 回滚和人工介入。

<h2 id="q-027">面试问题：Agent Guardrails 的边界如何快速概括？</h2>

Guardrails 应覆盖四类边界：

1. **输入边界**

   检测恶意请求、越权请求、敏感数据、注入攻击。

2. **工具边界**

   检查工具参数、权限、资源范围、是否需要用户确认。

3. **输出边界**

   检查最终回答是否包含敏感信息、违规内容、虚假承诺。

4. **状态边界**

   防止错误记忆写入、敏感数据进入长期记忆、跨用户数据污染。

Guardrails 可以是规则、分类器、小模型、策略引擎或人工审批。

<h2 id="q-028">面试问题：Agent 可观测性和 tracing 应该记录什么？</h2>

Agent tracing 应记录：

- 用户输入。
- 系统指令版本。
- 模型调用参数。
- 每次模型输出。
- 工具调用名称、参数、结果、耗时。
- handoff 事件。
- guardrail 结果。
- 权限审批记录。
- token、成本、延迟。
- 错误和重试。
- 最终输出。

可观测性的目标：

- Debug 为什么选错工具。
- 复现失败案例。
- 统计成本和延迟。
- 做安全审计。
- 构造评测数据集。
- 持续优化 prompt、工具和策略。

<h2 id="q-029">面试问题：Agent 常用评测基准有哪些？</h2>

| 基准 | 评估重点 |
| --- | --- |
| SWE-bench / SWE-bench Verified | 真实 GitHub issue 修复能力 |
| WebArena | 浏览器环境中的网页任务执行 |
| OSWorld | 操作系统/GUI 任务执行 |
| GAIA | 需要工具和多步推理的通用助理任务 |
| AgentBench | 多环境 Agent 能力 |
| τ-bench | 工具-用户-Agent 多轮交互和规则遵循 |
| HumanEval/MBPP | 代码生成基础能力 |
| 自建业务评测 | 企业流程、API 状态变更、规则合规 |

生产评估不要只看一次成功率，还要看：

- pass@k 或多次运行稳定性。
- 工具调用错误率。
- 任务完成时间。
- 成本。
- 安全拒绝准确率。
- 人工接管率。
- 回滚率。

<h2 id="q-030">面试问题：如何设计一个企业级 AI Agent 系统？</h2>

这里给出的是总览型答题框架，适合面试开场快速搭骨架；企业级 Agent Builder、Runtime、模型网关、Registry、多租户、Marketplace、平台选型等展开内容，见 [07_企业级Agent平台与产品落地高频考点.md](07_企业级Agent平台与产品落地高频考点.md)。

可以按以下框架回答：

1. **场景定义**

   明确是客服、数据分析、研发提效、运营自动化、企业搜索还是流程审批。

2. **能力边界**

   定义 Agent 能做什么、不能做什么、哪些动作需要人工确认。

3. **模型与路由**

   简单任务用便宜模型，复杂规划用强模型，敏感任务走人工。

4. **工具和协议**

   用 Function Calling 接业务 API，用 MCP 接标准工具，用 A2A 接远程专用 Agent。

5. **状态和记忆**

   会话状态、任务状态、长期记忆、企业知识库分开管理。

6. **工作流编排**

   确定性流程用 workflow，开放任务用 Agent，复杂流程用状态图。

7. **安全与权限**

   RBAC、审批、沙箱、审计、数据脱敏、prompt injection 防护。

8. **评估与观测**

   离线任务集、线上 tracing、人工反馈、失败样本回流。

9. **上线策略**

   从只读建议模式开始，再开放低风险工具，最后逐步开放写操作。

---

## 高频速记

1. Agent = LLM + Tools + Memory + Planning + Control Loop + Environment。
2. Workflow 适合确定性流程，Agent 适合开放式目标驱动任务。
3. ReAct 简单直接，Plan-and-Execute 适合长任务，Reflection 适合纠错。
4. Function Calling 是工具调用能力，MCP 是工具和上下文生态协议。
5. MCP 连接工具，A2A 连接 Agent。
6. OpenAI Agents SDK 强调 handoff、guardrails、tracing、MCP。
7. LangGraph 适合有状态、可恢复、可人工介入的复杂流程。
8. 编码 Agent 是 Agent 最成熟的落地场景之一。
9. Memory 解决连续性，RAG 解决外部知识检索。
10. 企业级 Agent 必须设计权限、沙箱、审计、评测和回滚。

## 参考资料

- Yao et al., [**ReAct: Synergizing Reasoning and Acting in Language Models**](https://arxiv.org/abs/2210.03629), 2022.
- Anthropic, [**Model Context Protocol**](https://modelcontextprotocol.io/).
- OpenAI, [**Agents SDK Documentation**](https://openai.github.io/openai-agents-python/).
- LangChain, [**LangGraph Documentation**](https://langchain-ai.github.io/langgraph/).
- Google, [**Agent Development Kit and A2A Protocol**](https://google.github.io/adk-docs/a2a/).
- agentsmd, [**AGENTS.md: A Simple Open Format for Guiding Coding Agents**](https://github.com/agentsmd/agents.md).
- Yao et al., [**tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains**](https://arxiv.org/abs/2406.12045), 2024.
