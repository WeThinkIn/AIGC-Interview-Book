# 目录

## 第一章 Agent 协议标准化总览

[1. 为什么 AIGC 时代需要 Agent 协议标准化？](#q-001)
  - [面试问题：Function Calling、MCP、A2A 分别解决什么问题？](#q-002)
  - [面试问题：为什么说 MCP 像 Agent 工具生态的 USB-C？](#q-003)
  - [面试问题：为什么说 A2A 解决的是 Agent 之间的互操作？](#q-004)

## 第二章 Function Calling 与工具调用基础

[2. Function Calling 的本质是什么？](#q-005)
  - [面试问题：Function Calling 和传统 API 调用有什么区别？](#q-006)
  - [面试问题：工具 Schema 应该如何设计？](#q-007)
  - [面试问题：大模型如何获得 Function Calling 能力？](#q-008)
  - [面试问题：Function Calling 为什么还不等于完整 Agent？](#q-009)

## 第三章 MCP 核心架构与能力

[3. MCP 的核心架构是什么？](#q-010)
  - [面试问题：MCP Host、Client、Server 分别负责什么？](#q-011)
  - [面试问题：MCP Tools、Resources、Prompts 有什么区别？](#q-012)
  - [面试问题：Roots、Sampling、Elicitation 分别解决什么问题？](#q-013)
  - [面试问题：MCP 的 stdio 与 Streamable HTTP 如何选择？](#q-014)

## 第四章 MCP 工程设计与安全

[4. 如何设计一个高质量 MCP Server？](#q-015)
  - [面试问题：MCP Server 的工具粒度应该如何划分？](#q-016)
  - [面试问题：MCP 如何处理认证、授权和凭证安全？](#q-017)
  - [面试问题：MCP 的 Prompt Injection 风险在哪里？](#q-018)
  - [面试问题：企业内部 MCP Registry / Tool Gateway 应该如何设计？](#q-019)

## 第五章 A2A 核心概念与协作机制

[5. A2A 协议的核心架构是什么？](#q-020)
  - [面试问题：Agent Card 的作用是什么？](#q-021)
  - [面试问题：A2A 中 Message、Task、Part、Artifact 如何理解？](#q-022)
  - [面试问题：A2A 如何支持流式响应和异步通知？](#q-023)
  - [面试问题：A2A 适合哪些多 Agent 协作场景？](#q-024)

## 第六章 MCP、A2A 与企业落地

[6. MCP 和 A2A 如何组合成企业 Agent 生态？](#q-025)
  - [面试问题：MCP、A2A、OpenAPI、LangGraph、Agent SDK 如何协同？](#q-026)
  - [面试问题：什么时候不应该使用 MCP 或 A2A？](#q-027)
  - [面试问题：如何回答“请设计一个企业级 Agent 协议平台”？](#q-028)

---

<h1 id="q-001">1. 为什么 AIGC 时代需要 Agent 协议标准化？</h1>

AIGC 时代的 Agent 不再只是聊天窗口，而是需要连接工具、数据、服务、其他 Agent 和真实业务系统。没有标准协议时，每个应用都要单独适配：

- 模型如何发现工具？
- 工具参数如何描述？
- 工具结果如何返回？
- 文件、数据库、知识库如何作为上下文暴露？
- 多个 Agent 如何互相发现、发消息、交接任务？
- 权限、认证、审计、流式响应如何统一？

如果每个模型应用、每个工具、每个 Agent 都用私有协议，生态会变成大量孤岛。协议标准化的价值是把“模型能力”“工具能力”“Agent 服务能力”解耦，让不同组件可以被复用、治理和替换。

可以用三层来理解：

| 层级 | 代表能力 | 解决的问题 |
| --- | --- | --- |
| Function Calling / Tool Use | 单个模型调用应用内工具 | 模型如何调用函数 |
| MCP | Agent 应用连接外部工具、资源、提示模板 | 工具和上下文如何标准化接入 |
| A2A | Agent 服务之间通信与协作 | Agent 如何互相发现、请求、交付结果 |

面试中可以总结：Function Calling 是工具调用能力，MCP 是工具和上下文接入协议，A2A 是 Agent 服务互操作协议。

<h2 id="q-002">面试问题：Function Calling、MCP、A2A 分别解决什么问题？</h2>

| 技术 | 主要对象 | 核心问题 | 典型例子 |
| --- | --- | --- | --- |
| Function Calling | 模型与应用内函数 | 模型如何输出结构化函数名和参数 | 查询天气、调用订单 API |
| MCP | Agent 应用与工具/数据源 | 如何用标准方式接入外部工具、资源和上下文 | GitHub、数据库、文件系统、浏览器、知识库 |
| A2A | Agent 与 Agent | 独立 Agent 服务如何发现能力、发送任务、返回结果 | 研究 Agent 调用财务分析 Agent |

三者之间不是替代关系，而是递进关系：

1. Function Calling 让模型能“调用函数”。
2. MCP 让工具和上下文能“以标准服务形式被调用”。
3. A2A 让一个 Agent 能“把任务交给另一个 Agent 服务”。

一个企业级 Agent 系统中，三者经常同时存在：

- 用 Function Calling 调应用内业务函数。
- 用 MCP 接企业知识库、数据库、代码仓库和浏览器。
- 用 A2A 调用其他部门或其他平台提供的专用 Agent。

<h2 id="q-003">面试问题：为什么说 MCP 像 Agent 工具生态的 USB-C？</h2>

MCP 的目标是为模型应用连接外部上下文和工具提供统一接口。它像 USB-C 的地方在于：

- 应用不需要为每个工具写一套私有集成。
- 工具提供方可以按统一协议暴露能力。
- 同一个 MCP Server 可以被不同 Agent Host 复用。
- 工具、资源、提示模板可以被动态发现。
- 本地工具和远程服务可以采用统一交互模型。

没有 MCP 时：

$$
\text{Agent App} \times \text{Tool Provider} = N \times M \text{ 个私有连接器}
$$

有 MCP 后：

$$
\text{Agent Host} \leftrightarrow \text{MCP Client} \leftrightarrow \text{MCP Server} \leftrightarrow \text{Tool/Data Source}
$$

当然，MCP 不是万能的。它主要标准化“工具和上下文接入”，不负责所有业务权限、工作流编排和多 Agent 协作。实际落地时还需要 RBAC、审计、沙箱、评测、模型网关和业务系统治理。

<h2 id="q-004">面试问题：为什么说 A2A 解决的是 Agent 之间的互操作？</h2>

A2A 关注的是独立 Agent 服务之间如何通信。一个 Agent 可能由不同团队、不同框架、不同模型、不同部署环境实现，但只要遵循 A2A，就可以用统一方式暴露能力和处理任务。

A2A 解决的问题包括：

- Agent 如何声明自己能做什么。
- Client Agent 如何发现 Remote Agent 的能力。
- 如何向远程 Agent 发送消息或任务。
- 远程 Agent 如何返回中间状态和最终结果。
- 如何处理多模态内容、文件和结构化结果。
- 长任务如何流式输出或异步通知。
- 用户交互和人工补充信息如何进入任务流程。

需要纠正一个常见误解：A2A 不是区块链共识协议，也不是 PBFT 这类分布式一致性算法。它更像 Agent 服务之间的应用层互操作协议，重点是能力发现、任务通信、状态管理、流式响应和结果交付。

---

<h1 id="q-005">2. Function Calling 的本质是什么？</h1>

Function Calling 的本质是让模型根据自然语言需求，选择一个结构化函数，并生成符合 Schema 的参数，由宿主程序执行函数，再把结果交回模型生成最终回答。

基本流程：

$$
\text{User Request} \rightarrow \text{Model Selects Function} \rightarrow \text{Generate Arguments} \rightarrow \text{Host Executes} \rightarrow \text{Tool Result} \rightarrow \text{Model Response}
$$

关键点：

- 模型不直接执行函数。
- 模型只生成函数名和参数。
- 宿主程序负责真正调用 API、数据库、文件系统或业务服务。
- 工具返回结果后，模型再负责解释、总结或决定下一步。

Function Calling 是 Agent 的基础能力之一，但完整 Agent 还需要记忆、规划、工具选择、权限、安全、状态管理和评估。

<h2 id="q-006">面试问题：Function Calling 和传统 API 调用有什么区别？</h2>

| 维度 | 传统 API 调用 | Function Calling |
| --- | --- | --- |
| 触发方式 | 开发者在代码中显式调用 | 模型根据语义选择工具 |
| 输入来源 | 程序构造结构化参数 | 模型从自然语言中抽取参数 |
| 工具选择 | 固定逻辑或规则 | 模型结合上下文动态判断 |
| 输出处理 | 程序直接消费 API 结果 | 模型将结果转成自然语言或下一步行动 |
| 错误恢复 | 程序异常处理 | 可结合 Agent 反思、重试、换工具 |
| 风险 | 主要是业务异常 | 还包含误调用、参数幻觉、越权调用 |

传统 API 调用是“程序调用服务”；Function Calling 是“模型建议调用哪个服务以及如何调用，程序负责执行”。

面试中要强调：Function Calling 把自然语言和结构化业务能力连接起来，但真正的执行权仍然应该掌握在宿主系统和权限策略中。

<h2 id="q-007">面试问题：工具 Schema 应该如何设计？</h2>

高质量工具 Schema 需要让模型清楚知道“什么时候用、怎么用、不能怎么用”。

设计要点：

1. **工具名清晰**

   使用具体动词和对象，例如 `search_customer_orders` 比 `query` 更好。

2. **描述强调适用边界**

   不只写“查询订单”，还要写“用于根据用户 ID 查询历史订单，不用于创建或修改订单”。

3. **参数类型明确**

   使用枚举、范围、必填字段、格式约束，减少模型自由发挥。

4. **避免万能工具**

   一个 `execute_sql` 或 `run_shell` 可以做很多事，但风险很高。生产场景应封装成更细粒度业务工具。

5. **返回结构稳定**

   返回结果应包含状态码、错误原因、关键字段、可展示摘要，便于模型理解。

6. **错误可恢复**

   错误信息要说明是参数缺失、权限不足、资源不存在、超时还是系统异常。

7. **安全边界内置**

   Schema 不能代替权限系统。危险参数、越权资源、敏感字段必须由宿主系统拦截。

反例：

```json
{
  "name": "do_action",
  "description": "执行操作",
  "parameters": {
    "input": "string"
  }
}
```

这个 Schema 太模糊，会导致模型不知道何时调用、如何构造参数，也不利于审计。

<h2 id="q-008">面试问题：大模型如何获得 Function Calling 能力？</h2>

Function Calling 能力通常来自多阶段训练和产品工程配合：

1. **预训练**

   模型学习大量代码、JSON、API 文档和自然语言结构，为格式化输出打基础。

2. **监督微调**

   使用带工具描述、用户请求、函数选择、参数生成的样本，让模型学会根据语义选择工具。

3. **偏好优化**

   通过人类或自动评估反馈，让模型更倾向于正确工具、正确参数和合适的拒绝。

4. **工具调用运行时**

   模型 API 或 Agent Runtime 提供 tool schema、tool choice、parallel tool calls、structured output、retry 等工程能力。

5. **在线评测与纠错**

   用真实日志评估工具选择准确率、参数正确率、调用成功率和安全违规率。

训练数据通常包含：

- 用户自然语言请求。
- 可用工具列表。
- 工具描述和参数 Schema。
- 正确工具选择。
- 正确参数 JSON。
- 工具返回结果。
- 最终自然语言回答。
- 不应调用工具的负样本。

面试中可以说：Function Calling 不是“模型天生会调用工具”，而是模型能力、结构化输出训练、API 运行时和工具治理共同形成的结果。

<h2 id="q-009">面试问题：Function Calling 为什么还不等于完整 Agent？</h2>

Function Calling 只解决“如何调用一个函数”这一局部问题。完整 Agent 还需要：

- 目标理解。
- 任务分解。
- 多步规划。
- 多工具选择。
- 工具结果解释。
- 失败恢复。
- 记忆和上下文管理。
- 权限审批。
- 长任务状态。
- 人机协作。
- 可观测性和评测。

对比：

| 能力 | Function Calling | Agent |
| --- | --- | --- |
| 单次工具调用 | 支持 | 支持 |
| 多步循环 | 不一定 | 核心能力 |
| 任务状态 | 通常无 | 必须有 |
| 记忆 | 不负责 | 常见核心模块 |
| 规划 | 不负责 | 常见核心模块 |
| 权限和审计 | 依赖宿主 | 必须系统化 |
| 自主性 | 低到中 | 中到高 |

一句话：Function Calling 是 Agent 的工具接口，Agent 是围绕目标、状态、工具、记忆和反馈构建的行动系统。

---

<h1 id="q-010">3. MCP 的核心架构是什么？</h1>

MCP，全称 Model Context Protocol，是一种开放协议，用来标准化模型应用如何向大模型提供上下文，以及如何连接外部工具和数据源。

MCP 的经典架构包括：

$$
\text{MCP Host} \rightarrow \text{MCP Client} \rightarrow \text{MCP Server}
$$

其中：

- Host 是用户实际使用的 AI 应用。
- Client 是 Host 内部维护的协议连接。
- Server 是暴露工具、资源和提示模板的服务。

MCP Server 可以连接：

- 本地文件系统。
- Git 仓库。
- 数据库。
- 浏览器。
- 企业知识库。
- 设计工具。
- 云服务。
- 业务 API。
- 内部搜索和日志系统。

它的核心价值是把工具生态从“每个应用各自集成”变成“标准服务器可复用”。

<h2 id="q-011">面试问题：MCP Host、Client、Server 分别负责什么？</h2>

| 角色 | 职责 | 例子 |
| --- | --- | --- |
| Host | 面向用户的 AI 应用，负责模型、UI、权限和上下文总控 | IDE Agent、桌面 Agent、聊天应用 |
| Client | Host 内部的连接实例，负责与一个 Server 通信 | 每接一个 MCP Server 就有一个 Client |
| Server | 暴露工具、资源、提示模板和能力声明 | GitHub Server、Postgres Server、File Server |

交互流程：

1. Host 启动或连接 MCP Server。
2. Client 与 Server 初始化连接并协商能力。
3. Host 发现 Server 暴露的 Tools、Resources、Prompts。
4. 模型在需要时选择工具或读取资源。
5. Client 把请求发给 Server。
6. Server 执行操作或返回上下文。
7. Host 把结果放回模型上下文。

注意：MCP Server 不等于模型服务。它通常不负责推理，而是给模型应用提供外部能力。

<h2 id="q-012">面试问题：MCP Tools、Resources、Prompts 有什么区别？</h2>

MCP 的三类核心服务能力可以这样理解：

| 能力 | 类比 | 主要用途 | 控制方 |
| --- | --- | --- | --- |
| Tools | 可执行函数 | 执行动作、查询系统、调用 API | 模型通常可选择调用 |
| Resources | 可读取资料 | 暴露文件、记录、数据库内容、上下文片段 | 应用通常选择加载 |
| Prompts | 可复用提示模板 | 提供任务模板、工作流模板、专家提示 | 用户或应用选择使用 |

具体区别：

- **Tools** 有副作用风险，例如写文件、发消息、查数据库、创建工单。
- **Resources** 更像上下文数据源，通常用于读取，不应默认有副作用。
- **Prompts** 是可复用的提示模板，帮助用户或 Host 快速构造任务。

设计时不要把所有东西都做成 Tool。比如“读取项目 README”更适合 Resource；“生成代码审查任务模板”更适合 Prompt；“创建 GitHub Issue”才是 Tool。

<h2 id="q-013">面试问题：Roots、Sampling、Elicitation 分别解决什么问题？</h2>

除了 Tools、Resources、Prompts，MCP 还包含一些容易被忽略但很重要的能力：

| 能力 | 解决的问题 | 例子 |
| --- | --- | --- |
| Roots | Host 告诉 Server 当前可访问的边界 | 当前工作区、项目根目录、允许读取的路径 |
| Sampling | Server 请求 Host 代为调用模型 | MCP Server 需要模型帮助总结或推理 |
| Elicitation | Server 请求 Host 向用户补充信息 | 工具执行前需要用户选择账号或确认参数 |

Roots 的意义是边界控制。文件类 Server 不应该默认扫描整台机器，而应该只在 Host 提供的 roots 内工作。

Sampling 的意义是让 Server 在需要模型能力时不必自己持有模型密钥，而是通过 Host 受控请求模型。

Elicitation 的意义是把“工具执行中需要用户补充信息”变成标准交互，而不是让 Server 私下弹窗或阻塞。

面试中可以说：Tools / Resources / Prompts 是 MCP 的常用表层能力，Roots / Sampling / Elicitation 则体现了 MCP 对安全边界、模型调用权和用户交互权的工程化考虑。

<h2 id="q-014">面试问题：MCP 的 stdio 与 Streamable HTTP 如何选择？</h2>

新版 MCP 中，标准传输主要是 `stdio` 和 `Streamable HTTP`。旧的 HTTP+SSE 传输用于兼容历史实现，通常不建议新项目优先选择。

| 传输 | 工作方式 | 适合场景 | 注意事项 |
| --- | --- | --- | --- |
| stdio | Host 启动本地子进程，通过 stdin/stdout 交换 JSON-RPC 消息 | 本地开发工具、CLI、文件系统、低延迟单用户场景 | 权限继承宿主进程，需限制可执行来源 |
| Streamable HTTP | Server 作为 HTTP 服务，支持远程连接和流式响应 | 企业服务、云端工具、多用户部署、跨网络访问 | 必须做认证、授权、TLS、审计和限流 |
| SSE | 旧版 HTTP+Server-Sent Events 路线 | 兼容旧 MCP Server | 新项目优先迁移到 Streamable HTTP |

选择建议：

- 本地私人工具优先 `stdio`。
- 团队共享或云端工具优先 `Streamable HTTP`。
- 远程 MCP Server 不应裸露公网。
- 本地 HTTP Server 应默认绑定 `localhost`。
- 对敏感工具要加 RBAC、审计和参数级权限控制。

需要注意，不能简单说 `stdio` 永远最好。它低延迟、部署简单，但不适合多租户、远程服务和集中治理。

---

<h1 id="q-015">4. 如何设计一个高质量 MCP Server？</h1>

高质量 MCP Server 不只是“把 API 包一层”。它要让模型能正确发现、理解、调用和恢复错误，同时满足权限、安全和可观测性要求。

设计清单：

- 能力边界清晰。
- Tool / Resource / Prompt 分工合理。
- 工具 Schema 精确。
- 错误信息结构化。
- 支持认证和授权。
- 参数级权限控制。
- 输出脱敏。
- 支持审计日志。
- 支持超时、重试和限流。
- 对危险操作加入确认或 dry-run。
- 与 Host 的 Roots、Sampling、Elicitation 协同。

面试中可以说：好的 MCP Server 是面向 Agent 的产品化工具服务，不是随便把内部 API 暴露给模型。

<h2 id="q-016">面试问题：MCP Server 的工具粒度应该如何划分？</h2>

工具粒度要在“过细”和“过粗”之间平衡。

过细的问题：

- 模型需要多次调用才能完成简单任务。
- 调用链变长，延迟和失败率上升。
- 模型容易选错相近工具。

过粗的问题：

- 工具像黑箱，模型难以控制。
- 参数复杂，Schema 难以写清。
- 权限难以精细化。
- 副作用风险集中。

推荐原则：

- 一个 Tool 对应一个清晰业务意图。
- 读写分离，例如 `search_issues` 和 `create_issue` 分开。
- 查询类工具支持过滤、分页和字段选择。
- 修改类工具支持 dry-run 或 preview。
- 高风险工具需要显式确认字段，例如 `confirm=true` 不能由模型自动填充。
- 对频繁组合使用的步骤，可以封装为安全的复合工具。

例如数据库 MCP 不应该只暴露 `execute_sql`，而应优先暴露 `list_tables`、`describe_table`、`query_readonly`、`run_approved_report` 等更可控工具。

<h2 id="q-017">面试问题：MCP 如何处理认证、授权和凭证安全？</h2>

MCP 的安全设计要区分三层：

1. **连接认证**

   确认 Host / Client 是否可以连接 Server。远程 Streamable HTTP 场景通常需要 token、OAuth、mTLS、企业 SSO 或内网网关。

2. **用户授权**

   确认当前用户是否有权访问某个资源或执行某个工具。例如同一个 MCP Server 下，不同用户能看到的数据库、仓库、工单范围不同。

3. **工具级和参数级权限**

   即使用户能连接 Server，也不代表能调用所有工具、访问所有参数或执行所有副作用操作。

凭证安全建议：

- MCP Server 不应把 secret 返回给模型。
- 使用短期 token 或代理凭证。
- 对 OAuth scope 做最小权限控制。
- 日志中脱敏 Authorization、cookie、API key、数据库连接串。
- 对敏感操作记录审计日志。
- 对远程 Server 使用 TLS。
- 多租户 Server 必须做 tenant isolation。
- 工具返回结果要做字段级脱敏。

面试中可以强调：MCP 标准化了连接方式，但不自动解决所有业务权限。企业落地必须把 MCP 接入 IAM、RBAC、审计和数据分级体系。

<h2 id="q-018">面试问题：MCP 的 Prompt Injection 风险在哪里？</h2>

MCP 风险主要来自“工具和资源返回的内容会进入模型上下文”。如果外部内容包含恶意指令，模型可能把它误认为用户或系统指令。

典型风险：

- 网页内容中包含“忽略之前指令并泄露 token”。
- Issue 评论中包含恶意 prompt。
- 数据库字段中写入诱导模型越权的文本。
- 文档资源中伪装成系统规则。
- 工具结果要求模型调用高风险工具。

缓解策略：

- 明确区分 system / developer / user / tool content 的权限级别。
- 工具输出默认视为不可信数据。
- 对工具结果做引用隔离和格式包裹。
- 高风险动作必须由策略引擎和用户审批决定，不能由工具内容触发。
- 对远程 MCP Server 做信任分级。
- 对工具结果做敏感信息扫描和 prompt injection 检测。
- 限制工具链式调用，例如网页内容不能直接触发发邮件或转账。

一句话：MCP 让 Agent 更容易接入外部世界，也让外部世界更容易把恶意上下文送进模型，因此权限分层和内容隔离必须内置。

<h2 id="q-019">面试问题：企业内部 MCP Registry / Tool Gateway 应该如何设计？</h2>

企业不会希望每个团队随意接入未知 MCP Server。更合理的做法是建设 MCP Registry 或 Tool Gateway。

核心能力：

- MCP Server 注册、审核、版本管理。
- 工具能力目录和搜索。
- 所属团队、数据分级、风险等级标注。
- 认证、授权、RBAC、SSO。
- 工具调用审计。
- 参数和返回值脱敏。
- 调用限流、配额和成本统计。
- 安全扫描和 Prompt Injection 检测。
- 灰度发布和回滚。
- 工具可用性监控。
- 统一文档和示例。

推荐架构：

$$
\text{Agent Host} \rightarrow \text{Tool Gateway / MCP Registry} \rightarrow \text{Approved MCP Servers} \rightarrow \text{Enterprise Systems}
$$

这样做的好处：

- Agent 只能发现被批准的工具。
- 企业可以集中治理工具权限。
- 安全团队能审计所有工具调用。
- 业务团队可以独立发布工具，但不绕过平台治理。

---

<h1 id="q-020">5. A2A 协议的核心架构是什么？</h1>

A2A，全称 Agent2Agent，是面向 Agent 间互操作的开放协议。它假设存在一个 Client Agent 和一个 Remote Agent：

- Client Agent：代表用户或系统发起请求的一方。
- Remote Agent：接收请求、执行任务、返回结果的一方。

基本流程：

1. Client 通过 Agent Card 发现 Remote Agent 的能力。
2. Client 向 Remote Agent 发送 Message 或创建 Task。
3. Remote Agent 处理任务，必要时请求补充输入。
4. Remote Agent 返回状态更新、流式事件或最终 Artifact。
5. Client 将结果呈现给用户或继续交给其他 Agent。

A2A 的重点不是让所有 Agent 共享内部实现，而是让它们通过统一协议交换任务和结果。Remote Agent 可以隐藏自己的模型、工具、记忆和内部规划，只暴露可调用能力和结果。

<h2 id="q-021">面试问题：Agent Card 的作用是什么？</h2>

Agent Card 是 Remote Agent 的能力说明书，类似服务发现文档。

它通常描述：

- Agent 名称和描述。
- 服务端点 URL。
- 支持的认证方式。
- 支持的输入输出模式。
- 支持的能力，例如 streaming、push notification、state transition history。
- skills 或具体能力列表。
- 版本、提供方、文档链接。

Agent Card 的价值：

- Client 可以发现 Remote Agent 能做什么。
- 上层 Router 可以根据能力选择合适 Agent。
- 企业可以构建 Agent Registry。
- 可以在调用前判断认证和模态是否匹配。

面试中可以类比：OpenAPI 描述 API 服务能力，Agent Card 描述 Agent 服务能力。

<h2 id="q-022">面试问题：A2A 中 Message、Task、Part、Artifact 如何理解？</h2>

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| Message | 一次通信消息 | 用户请求、Agent 追问、状态说明 |
| Task | 一个需要跟踪状态的工作单元 | “分析这份财报并生成摘要” |
| Part | Message 或 Artifact 中的内容片段 | text part、file part、data part |
| Artifact | 任务产出的结果对象 | 报告、图表、代码文件、结构化 JSON |

可以这样理解：

- Message 是沟通。
- Task 是任务容器。
- Part 是内容的基本组成。
- Artifact 是可交付成果。

为什么需要这些抽象？

- Agent 协作不只是发一句文本。
- 任务可能持续很久，需要状态。
- 输入输出可能是多模态的。
- 最终结果可能包含多个文件或结构化对象。
- Client 需要区分中间消息和最终交付物。

<h2 id="q-023">面试问题：A2A 如何支持流式响应和异步通知？</h2>

Agent 任务经常不是一次请求马上返回，尤其是研究、代码生成、数据分析、浏览器自动化等场景。A2A 因此需要支持长任务。

常见机制：

1. **Streaming**

   Remote Agent 在任务执行过程中持续发送状态、文本片段、工具进度或中间结果。

2. **Task Status**

   Task 可以有 submitted、working、input-required、completed、failed、canceled 等状态，Client 能跟踪进度。

3. **Input Required**

   当 Remote Agent 需要用户补充信息时，可以把任务状态切到需要输入，而不是直接失败。

4. **Push Notification**

   对长时间任务，Remote Agent 可以在完成后通过推送通知 Client，而不是要求 Client 一直保持连接。

5. **Artifact Delivery**

   最终结果以 Artifact 形式交付，支持文本、文件、结构化数据等多种 Part。

面试中可以说：A2A 的长任务能力使它更像 Agent 服务协议，而不是简单聊天 API。

<h2 id="q-024">面试问题：A2A 适合哪些多 Agent 协作场景？</h2>

A2A 适合“跨系统、跨团队、跨框架”的 Agent 协作：

- 企业内不同部门 Agent 互调。
- 一个总控 Agent 调用专业分析 Agent。
- 跨供应商 Agent 服务集成。
- 多模态 Agent 和业务 Agent 协作。
- 研究 Agent 调用代码 Agent、数据 Agent、文档 Agent。
- 客服 Agent 调用工单 Agent、订单 Agent、物流 Agent。

不一定需要 A2A 的场景：

- 单应用内部的多个函数。
- 同一个进程里的子 Agent。
- 固定 DAG 工作流。
- 只是为了调用数据库或文件系统。

这些场景用 Function Calling、MCP 或内部工作流可能更简单。

一句话：如果你要接的是工具，用 MCP；如果你要接的是另一个独立 Agent 服务，用 A2A。

---

<h1 id="q-025">6. MCP 和 A2A 如何组合成企业 Agent 生态？</h1>

本章只从协议层回答企业落地：工具如何接入、Agent 如何互操作、协议网关如何治理。完整企业 Agent 平台的 Builder、Runtime、模型网关、多租户、产品落地和选型，见 [07_企业级Agent平台与产品落地高频考点.md](07_企业级Agent平台与产品落地高频考点.md)。

企业级 Agent 生态可以按三类能力建设：

1. **工具生态**

   用 MCP 把数据库、知识库、代码仓库、浏览器、业务系统、日志平台、工单系统标准化暴露给 Agent。

2. **Agent 生态**

   用 A2A 把不同团队、不同模型、不同框架构建的 Agent 服务纳入统一互操作体系。

3. **治理体系**

   用 IAM、RBAC、审计、Tool Gateway、Agent Registry、评测平台、模型网关和安全策略管理所有调用。

示意结构：

$$
\text{User} \rightarrow \text{Agent Host} \rightarrow
\begin{cases}
\text{MCP Client} \rightarrow \text{MCP Server} \rightarrow \text{Tools/Data} \\
\text{A2A Client} \rightarrow \text{Remote Agent} \rightarrow \text{Artifacts}
\end{cases}
$$

面试中可以给出一句高分回答：MCP 让 Agent 会使用工具，A2A 让 Agent 会委托其他 Agent，治理平台让这些能力可控、可审计、可规模化。

<h2 id="q-026">面试问题：MCP、A2A、OpenAPI、LangGraph、Agent SDK 如何协同？</h2>

| 技术 | 角色定位 |
| --- | --- |
| OpenAPI | 描述传统 HTTP API |
| Function Calling | 让模型选择函数并生成参数 |
| MCP | 把工具、资源、提示模板标准化暴露给 Agent Host |
| A2A | 让独立 Agent 服务之间互操作 |
| LangGraph | 用图状态机编排可持久化 Agent 工作流 |
| Agent SDK | 提供 Agent、Tool、Handoff、Guardrails、Tracing 等开发框架 |

协同方式：

- 业务系统已有 OpenAPI，可以包装成 Function Tool 或 MCP Tool。
- Agent Host 通过 MCP 接工具，通过 A2A 接其他 Agent。
- LangGraph 负责编排多步骤状态机。
- Agent SDK 负责模型调用、工具调用、handoff、guardrails 和 tracing。
- 企业平台统一做权限、审计、评测和发布。

一个实际例子：

1. 用户让办公 Agent 生成季度经营分析。
2. Agent 用 MCP 查询数据仓库和文档库。
3. Agent 用 A2A 调用财务分析 Agent 生成指标解释。
4. LangGraph 管理“取数 -> 分析 -> 生成报告 -> 人工审批 -> 发布”的状态。
5. Agent SDK 记录 tracing，并在敏感动作前触发 guardrail。

<h2 id="q-027">面试问题：什么时候不应该使用 MCP 或 A2A？</h2>

不应该为了追赶概念而滥用协议。

不适合 MCP 的情况：

- 只是应用内一个简单函数，直接 Function Calling 更轻。
- 工具不会被多个 Host 复用。
- 工具没有清晰权限边界。
- 暴露 MCP 会增加不必要的攻击面。
- 团队没有能力维护 Server 版本、安全和监控。

不适合 A2A 的情况：

- Agent 都在同一个进程里，内部函数调用即可。
- 只是固定流程节点，不需要独立 Agent 服务。
- 没有跨团队、跨框架、跨部署互操作需求。
- 延迟敏感，远程 Agent 调用成本过高。
- Remote Agent 的能力和责任边界不清。

选择原则：

- 简单函数用 Function Calling。
- 可复用工具和上下文服务用 MCP。
- 独立 Agent 服务互调用 A2A。
- 强流程控制用 Workflow / LangGraph。
- 高风险生产场景优先治理和权限，而不是优先接更多协议。

<h2 id="q-028">面试问题：如何回答“请设计一个企业级 Agent 协议平台”？</h2>

注意这里设计的是“协议平台”，不是完整 Agent 平台。回答重点应落在 MCP/A2A 接入、能力发现、认证授权、调用链审计和协议兼容；完整平台架构题见 [07_企业级Agent平台与产品落地高频考点.md](07_企业级Agent平台与产品落地高频考点.md)。

可以按六层回答：

1. **接入层**

   支持 Web、IDE、IM、API、Webhook、Cron 等入口。

2. **Agent Host / Runtime 层**

   管理模型调用、工具调用、上下文、session、流式响应、任务状态和 tracing。

3. **MCP 工具层**

   建设 MCP Registry 和 Tool Gateway，统一接入数据库、知识库、文件系统、Git、工单、日志、浏览器等工具。

4. **A2A Agent 层**

   建设 Agent Registry，通过 Agent Card 管理远程 Agent 能力、认证方式、输入输出模式和 SLA。

5. **编排层**

   使用工作流或图状态机编排复杂流程，支持人审、重试、补偿、断点恢复和任务队列。

6. **治理层**

   统一 IAM、RBAC、租户隔离、审计日志、敏感信息脱敏、Prompt Injection 防御、评测、灰度发布、成本控制和监控告警。

关键指标：

- 工具调用成功率。
- 参数正确率。
- 任务完成率。
- 平均延迟和成本。
- 人工接管率。
- 权限违规率。
- Prompt Injection 拦截率。
- Agent 间调用链可追踪性。

最后补一句：企业级 Agent 协议平台的核心不是“接入最多工具和 Agent”，而是让工具和 Agent 在统一协议、统一权限、统一审计和统一评测下可规模化运行。

---

## 高频速记

1. Function Calling 解决模型如何调用函数，MCP 解决工具和上下文如何标准化接入，A2A 解决 Agent 服务如何互操作。
2. MCP 的核心角色是 Host、Client、Server。
3. MCP Tools 负责动作，Resources 负责上下文数据，Prompts 负责可复用提示模板。
4. Roots 控制访问边界，Sampling 让 Server 请求 Host 代调模型，Elicitation 让 Server 请求用户补充信息。
5. MCP 标准传输主要是 stdio 和 Streamable HTTP，SSE 属于旧版兼容路线。
6. 远程 MCP 必须做认证、授权、TLS、审计、限流和租户隔离。
7. MCP 工具输出属于不可信内容，必须防 Prompt Injection。
8. A2A 的核心是 Client Agent、Remote Agent、Agent Card、Message、Task、Part、Artifact。
9. Agent Card 类似 Agent 服务的能力说明书。
10. A2A 支持长任务、流式状态、输入补充、异步通知和最终 Artifact。
11. 接工具用 MCP，接独立 Agent 服务用 A2A，应用内简单函数用 Function Calling。
12. 企业级 Agent 平台需要 MCP Registry、Tool Gateway、Agent Registry、工作流编排、权限审计和评测体系。

## 参考资料

- Model Context Protocol, [**Specification 2025-06-18**](https://modelcontextprotocol.io/specification/2025-06-18).
- Model Context Protocol, [**Transports**](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports).
- Model Context Protocol, [**Server Features**](https://modelcontextprotocol.io/specification/2025-06-18/server).
- Model Context Protocol, [**Client Features**](https://modelcontextprotocol.io/specification/2025-06-18/client).
- Model Context Protocol, [**Authorization**](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization).
- A2A Project, [**Agent2Agent Protocol Specification**](https://a2a-protocol.org/latest/specification/).
- OpenAI, [**Function Calling Guide**](https://platform.openai.com/docs/guides/function-calling).
- OpenAI, [**Agents SDK Documentation**](https://openai.github.io/openai-agents-python/).
