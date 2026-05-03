# 《三年面试五年模拟》之 AI Agent 基础知识高频考点

- :star: [01_AIGC时代AI Agent基础高频考点](01_AIGC时代AI Agent基础高频考点.md)
- :computer: [02_编码Agent与AgentOS工程高频考点](02_编码Agent与AgentOS工程高频考点.md)
- :triangular_ruler: [03_Agent设计模式与工作流编排高频考点](03_Agent设计模式与工作流编排高频考点.md)
- :link: [04_MCP与A2A协议高频考点](04_MCP与A2A协议高频考点.md)
- :brain: [05_Agent记忆与上下文工程高频考点](05_Agent记忆与上下文工程高频考点.md)
- :shield: [06_Agent安全评测与AgentOps高频考点](06_Agent安全评测与AgentOps高频考点.md)
- :office: [07_企业级Agent平台与产品落地高频考点](07_企业级Agent平台与产品落地高频考点.md)
- :test_tube: [08_Agent_Harness_Engineering高频考点](08_Agent_Harness_Engineering高频考点.md)
- :repeat: [09_自进化Agent与多平台运行时高频考点](09_自进化Agent与多平台运行时高频考点.md)

## 学习定位

本目录面向 AIGC 时代的 AI Agent、工具调用、MCP/A2A、多 Agent 协作、编码 Agent、上下文工程、记忆机制、权限安全和工程落地方向的算法与应用工程面试准备。

建议阅读顺序：

1. 先阅读 `01_AIGC时代AI Agent基础高频考点.md`，建立 Agent 技术地图，掌握 Agent、Workflow、MCP、A2A、Memory、RAG、评测和主流框架。
2. 再阅读 `02_编码Agent与AgentOS工程高频考点.md`，理解编码 Agent 与 AgentOS 工程系统的工具系统、权限模型、上下文压缩、Agent Runtime、Gateway、多通道、多 Agent、Sandbox、Hooks、后台任务和企业落地。
3. 然后阅读 `03_Agent设计模式与工作流编排高频考点.md`，系统掌握 ReAct、Plan-and-Solve、ReWOO、Reflection、Tree-of-Thought、Graph-of-Thought、LATS、多 Agent 编排和可持久化工作流。
4. 接着阅读 `04_MCP与A2A协议高频考点.md`，深入掌握 Function Calling、MCP Host/Client/Server、Tools/Resources/Prompts、Roots/Sampling/Elicitation、stdio/Streamable HTTP、A2A Agent Card、Task、Message、Artifact 和企业协议平台设计。
5. 继续阅读 `05_Agent记忆与上下文工程高频考点.md`，掌握 Memory 类型、Memory vs RAG、记忆生命周期、上下文窗口治理、Context Engine、跨 Agent 记忆复用、隐私安全和 Skills 渐进式上下文。
6. 再阅读 `06_Agent安全评测与AgentOps高频考点.md`，掌握 Agent Guardrails、权限沙箱、Prompt Injection 防护、Benchmark、Eval Harness、Tracing、灰度发布、事故响应和企业 AgentOps。
7. 接着阅读 `07_企业级Agent平台与产品落地高频考点.md`，掌握 Agent Builder、Runtime、模型网关、Tool/MCP/Agent Registry、多租户、私有化、Marketplace、低代码平台、产品落地和平台选型。
8. 然后阅读 `08_Agent_Harness_Engineering高频考点.md`，掌握 Agent Harness Engineering、任务数据、环境模拟、Runner、Trace、Grader、CI/CD 门禁、Shadow Mode、Model-native Harness 和企业级 Harness 平台设计。
9. 最后阅读 `09_自进化Agent与多平台运行时高频考点.md`，掌握自进化 Agent、多平台 Gateway、Provider Router、Tool Registry、Skill Hub、Plugin System、Curator、Checkpoint/Rollback、轨迹数据和 RL 环境。

## 章节职责

- `01_AIGC时代AI Agent基础高频考点.md`：总览型主线文档，只负责建立 Agent 技术地图和高频概念索引，不展开实现细节。
- `02_编码Agent与AgentOS工程高频考点.md`：工程型专题文档，负责将 Claude Code、OpenClaw 等项目中的通用架构提炼为编码 Agent / AgentOS 工程知识，重点是代码仓库、文件编辑、Shell、IDE/CLI、远程会话、AgentOS Runtime。
- `03_Agent设计模式与工作流编排高频考点.md`：模式型专题文档，负责 ReAct、规划执行、反思、搜索式推理、多 Agent 编排和可恢复工作流，不承担协议、安全平台和企业产品选型。
- `04_MCP与A2A协议高频考点.md`：协议型专题文档，负责 Function Calling、MCP、A2A、工具协议安全和协议平台设计，只从协议接入层讨论企业落地。
- `05_Agent记忆与上下文工程高频考点.md`：上下文型专题文档，负责 Agent Memory、RAG、Context Engine、上下文压缩、记忆存储、安全治理和 Skills 渐进式上下文，只在必要时用编码 Agent 举例。
- `06_Agent安全评测与AgentOps高频考点.md`：治理型专题文档，负责 Agent 安全、Guardrails、Benchmark、评测体系、Tracing、上线灰度、事故响应和 AgentOps，不重复企业平台产品架构，也不展开 Harness 工程实现。
- `07_企业级Agent平台与产品落地高频考点.md`：平台型专题文档，负责企业 Agent 平台架构、控制平面、Agent Builder、模型网关、多租户治理、产品落地和平台选型，只引用安全评测与协议能力，不重复底层原理。
- `08_Agent_Harness_Engineering高频考点.md`：评测工程型专题文档，负责 Agent Harness 的任务数据、环境模拟、运行器、trace、grader、回归门禁、CI/CD、Shadow Mode、编码/浏览器/客服/多 Agent Harness 和企业级 Harness 平台设计。
- `09_自进化Agent与多平台运行时高频考点.md`：长期运行型专题文档，负责从 Hermes Agent 等项目中提炼自进化 Agent、多平台 Gateway、Provider Router、Tool Registry、Skill Hub、Plugin System、Curator、自我维护、Checkpoint/Rollback、轨迹数据和 RL 环境等通用问答。

## 内容边界与去重原则

为了避免不同章节反复回答同一问题，本板块按“总览 -> 场景工程 -> 模式编排 -> 协议标准 -> 记忆上下文 -> 安全运营 -> 企业平台 -> Harness 工程 -> 自进化运行时”切分：

| 章节 | 只重点回答 | 深入内容归属 |
| --- | --- | --- |
| 01 | Agent 是什么、技术栈如何分层、常见概念如何识别 | 具体模式见 03，协议见 04，记忆见 05，安全评测见 06，企业平台见 07 |
| 02 | 编码 Agent 与 AgentOS 如何在真实工程环境中运行 | 通用安全治理见 06，通用记忆系统见 05，通用平台选型见 07，长期运行和自进化见 09 |
| 03 | Agent 如何规划、反思、搜索和多 Agent 编排 | 工具协议见 04，生产安全与评测见 06 |
| 04 | Function Calling、MCP、A2A 如何标准化连接工具和 Agent | 完整企业平台架构见 07，通用安全评测见 06 |
| 05 | Memory、RAG、Context Engine 如何管理信息生命周期 | 编码仓库工程细节见 02，安全运营见 06，自进化知识维护见 09 |
| 06 | Agent 如何评测、观测、上线、审计和响应事故 | 企业平台模块设计见 07，Harness 工程实现见 08 |
| 07 | 企业如何规模化建设、治理和运营 Agent 平台 | 协议细节见 04，记忆细节见 05，安全评测细节见 06，评测工程平台见 08 |
| 08 | Agent Harness 如何构造任务、环境、Runner、Trace、Grader 和发布门禁 | 安全策略和 AgentOps 理论见 06，企业平台整体架构见 07，轨迹训练和自进化闭环见 09 |
| 09 | 自进化 Agent 如何长期运行、跨平台交互、维护 Skill/Memory、沉淀轨迹数据 | 通用记忆细节见 05，安全评测见 06，企业平台架构见 07，Harness 工程见 08 |
