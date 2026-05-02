# 目录

## 第一章 编码 Agent 与 AgentOS 总览

[1. 编码 Agent 与 AgentOS 工程系统的核心价值是什么？](#q-001)
  - [面试问题：编码 Agent 与传统 IDE Copilot 有什么区别？](#q-002)
  - [面试问题：一个终端/IDE 编码 Agent 的典型架构是什么？](#q-003)
  - [面试问题：为什么编码 Agent 是 Agent 工程落地最快的场景？](#q-004)

## 第二章 工具系统与权限控制

[2. 编码 Agent 的工具系统如何设计？](#q-005)
  - [面试问题：FileRead、Edit、Bash、Search、Git、LSP 工具分别解决什么问题？](#q-006)
  - [面试问题：工具权限模型应该如何设计？](#q-007)
  - [面试问题：为什么 Bash 和文件写入工具必须强权限控制？](#q-008)
  - [面试问题：如何处理远程会话或云端执行中的权限请求？](#q-009)

## 第三章 上下文、记忆与项目知识

[3. 编码 Agent 如何理解一个真实代码仓库？](#q-010)
  - [面试问题：CLAUDE.md / AGENTS.md 这类项目记忆文件有什么作用？](#q-011)
  - [面试问题：上下文压缩、摘要和恢复在编码 Agent 中如何工作？](#q-012)
  - [面试问题：代码索引、ripgrep、LSP、文件缓存如何协同？](#q-013)
  - [面试问题：如何避免把无关文件塞满上下文？](#q-014)

## 第四章 子 Agent、Skills、Hooks 与任务编排

[4. 编码 Agent 为什么需要子 Agent 和 Skills？](#q-015)
  - [面试问题：子 Agent 适合处理哪些任务？](#q-016)
  - [面试问题：Skills 的本质是什么？与工具有什么区别？](#q-017)
  - [面试问题：Hooks 在编码 Agent 中有什么价值？](#q-018)
  - [面试问题：Plan Mode、Todo、任务状态如何提升长任务可靠性？](#q-019)

## 第五章 IDE/CLI/远程桥接与产品工程

[5. 编码 Agent 为什么常同时支持 CLI、IDE 和远程会话？](#q-020)
  - [面试问题：CLI Agent 和 IDE Agent 的体验差异是什么？](#q-021)
  - [面试问题：IDE Bridge 通常需要传递哪些信息？](#q-022)
  - [面试问题：远程编码 Agent 如何处理 session、interrupt、resume 和 reconnect？](#q-023)
  - [面试问题：MCP 在编码 Agent 中通常如何使用？](#q-024)

## 第六章 评测、可观测性与安全

[6. 如何评估编码 Agent 是否真的有用？](#q-025)
  - [面试问题：SWE-bench 类评测为什么重要但不够？](#q-026)
  - [面试问题：编码 Agent 的 tracing 和 telemetry 应该记录什么？](#q-027)
  - [面试问题：编码 Agent 常见失败模式有哪些？](#q-028)
  - [面试问题：如何设计企业内可用的编码 Agent？](#q-029)

## 第七章 AgentOS、Gateway 与多通道运行时

[7. AgentOS / 个人助理型 Agent 系统通常如何分层？](#q-030)
  - [面试问题：Gateway 控制平面在 Agent 系统中负责什么？](#q-031)
  - [面试问题：Provider、Model、Agent Runtime、Channel 如何解耦？](#q-032)
  - [面试问题：多通道 Agent 如何做会话路由和用户隔离？](#q-033)
  - [面试问题：多 Agent 系统如何做到 workspace、auth、session 隔离？](#q-034)

## 第八章 AgentOS 安全、上下文与自动化

[8. AgentOS 中 Sandbox、Tool Policy、Elevated 为什么要分开？](#q-035)
  - [面试问题：Context Engine 的 ingest / assemble / compact / after-turn 生命周期如何理解？](#q-036)
  - [面试问题：Hooks、Cron/Webhook、Background Task 在 AgentOS 中分别解决什么问题？](#q-037)
  - [面试问题：如何设计支持长任务、断线恢复和事件流的 Agent 运行时？](#q-038)

---

<h1 id="q-001">1. 编码 Agent 与 AgentOS 工程系统的核心价值是什么？</h1>

编码 Agent 与 AgentOS 工程系统的核心价值，是把大模型从“代码建议器”升级为“可受控的软件工程任务执行器”和“可长期运行的智能工作台”。Claude Code、Codex、Cursor、Devin、Jules、OpenClaw 等项目都可以作为案例来理解这一趋势，但面试中更重要的是抽象出通用架构能力，而不是记住某个产品的命令或术语。

这类系统不只是补全一段函数，而是可以围绕一个目标执行完整工程闭环：

$$
\text{理解需求} \rightarrow \text{读取代码} \rightarrow \text{制定计划} \rightarrow \text{编辑文件} \rightarrow \text{运行测试} \rightarrow \text{修复错误} \rightarrow \text{总结交付}
$$

编码 Agent 的关键能力包括：

- 仓库级代码理解。
- 文件读写和增量编辑。
- Shell 命令执行。
- 搜索、索引和 LSP 语义能力。
- Git diff、commit、PR、review。
- 项目规则和记忆加载。
- 权限审批和沙箱。
- 长任务上下文压缩和恢复。
- 子 Agent 并行探索。
- 工具调用 tracing 和成本统计。

<h2 id="q-002">面试问题：编码 Agent 与传统 IDE Copilot 有什么区别？</h2>

| 维度 | 传统 Copilot | 编码 Agent |
| --- | --- | --- |
| 交互方式 | 用户写代码，模型补全/建议 | 用户给目标，Agent 读写运行验证 |
| 上下文 | 当前文件或局部工程 | 仓库、历史、工具结果、项目记忆 |
| 行动能力 | 主要生成代码 | 可编辑文件、运行命令、查日志、提交变更 |
| 自主性 | 低，用户主导 | 中高，Agent 可规划多步 |
| 验证 | 用户运行测试 | Agent 可自行运行测试并迭代 |
| 风险 | 生成错误代码 | 可能执行危险命令或改错文件 |

一句话：

Copilot 更像副驾驶，编码 Agent 更像可以接手一张工单的初级工程执行者，但需要权限和审查。

<h2 id="q-003">面试问题：一个终端/IDE 编码 Agent 的典型架构是什么？</h2>

典型架构可以分为八层：

1. **入口层**

   CLI、IDE 插件、Web、SDK、远程 session。

2. **会话层**

   保存 messages、session id、用户配置、当前工作目录、模型选择、预算、权限模式。

3. **上下文层**

   收集系统提示词、项目说明、Git 状态、打开文件、选区、历史摘要、记忆。

4. **模型循环层**

   负责模型调用、streaming、tool-call loop、重试、token 统计、停止条件。

5. **工具层**

   文件读写、搜索、Bash、Git、LSP、Web、MCP、Notebook、子 Agent、Skills。

6. **权限层**

   判断工具是否允许、是否需要确认、是否被策略拒绝、是否运行在沙箱中。

7. **UI/交互层**

   展示 diff、命令输出、权限弹窗、进度、成本、任务列表和错误。

8. **观测和持久化层**

   记录 transcript、tool trace、session restore、cost、diagnostics、telemetry。

<h2 id="q-004">面试问题：为什么编码 Agent 是 Agent 工程落地最快的场景？</h2>

原因有五个：

1. **工具环境明确**

   代码仓库、文件系统、Shell、Git、测试框架天然适合工具调用。

2. **反馈可验证**

   编译、测试、lint、类型检查、CI 都能提供客观反馈。

3. **上下文结构化**

   代码文件、依赖、错误日志、diff、issue 都是相对结构化的信息。

4. **用户愿意审查**

   开发者可以 review diff，适合 human-in-the-loop。

5. **ROI 清晰**

   修 bug、写测试、重构、文档、代码审查都能直接节省工程时间。

---

<h1 id="q-005">2. 编码 Agent 的工具系统如何设计？</h1>

编码 Agent 的工具系统需要同时满足能力、可控和可观测。

一个工具通常包含：

- 工具名称。
- 用户可读描述。
- 输入 Schema。
- 权限规则。
- 执行逻辑。
- 进度事件。
- 结构化结果。
- 错误类型。
- 展示组件。

从现代编码 Agent / AgentOS 的工程形态看，工具不是简单函数，而是可被模型选择、可被系统审批、可被 UI 展示、可被日志审计的执行单元。

<h2 id="q-006">面试问题：FileRead、Edit、Bash、Search、Git、LSP 工具分别解决什么问题？</h2>

| 工具 | 作用 | 典型风险 |
| --- | --- | --- |
| FileRead | 读取源码、配置、日志、图片、Notebook | 上下文过长、敏感文件泄露 |
| Edit/Write | 修改或创建文件 | 改错文件、覆盖用户改动 |
| Bash | 运行测试、构建、脚本、Git 命令 | 危险命令、网络/系统副作用 |
| Search/Grep/Glob | 快速定位文件和符号 | 搜索范围过大、噪声多 |
| Git | 查看 diff、commit、branch、PR 状态 | 误提交、误切分支 |
| LSP | 跳转定义、引用、诊断、补全 | 依赖语言服务稳定性 |
| Web/MCP | 查文档、调用外部服务 | prompt injection、凭证泄露 |

设计原则：

- 读工具可以相对宽松。
- 写工具需要 diff 和确认。
- 执行工具要区分只读命令和危险命令。
- 远程/生产工具必须有认证和审计。

<h2 id="q-007">面试问题：工具权限模型应该如何设计？</h2>

本节只讨论编码 Agent 场景下的工具权限，例如 Bash、文件写入、Git、LSP、远程 worker 和 IDE Bridge。通用 Agent Guardrails、Prompt Injection、企业审计和上线治理，见 [06_Agent安全评测与AgentOps高频考点.md](06_Agent安全评测与AgentOps高频考点.md)。

权限模型建议包含：

1. **权限模式**

   - 默认模式：危险操作需确认。
   - 计划模式：只读和规划，不执行写操作。
   - 自动模式：低风险操作可自动执行。
   - 绕过模式：仅在受信任沙箱中使用。

2. **规则来源**

   - 系统默认规则。
   - 用户配置。
   - 项目配置。
   - 企业策略。
   - 临时本轮授权。

3. **工具级权限**

   - always allow。
   - always deny。
   - always ask。
   - 根据参数动态判断。

4. **环境边界**

   - 工作目录白名单。
   - 网络访问限制。
   - 文件系统沙箱。
   - 密钥和环境变量保护。

5. **审计记录**

   记录谁在什么时候允许了什么工具，以什么参数执行，产生了什么结果。

<h2 id="q-008">面试问题：为什么 Bash 和文件写入工具必须强权限控制？</h2>

Bash 和文件写入工具有真实副作用：

- 删除文件。
- 覆盖用户未保存工作。
- 修改配置。
- 泄露环境变量。
- 安装未知依赖。
- 访问网络。
- 提交或推送代码。
- 运行消耗资源的进程。

风险控制策略：

- 命令分类：只读、写入、网络、破坏性、长时间运行。
- 高风险命令强制确认。
- 禁止 `rm -rf`、`git reset --hard` 等除非用户明确要求。
- 写文件前检查是否有用户未保存或未归属改动。
- 使用 diff 展示变更。
- 对命令输出做脱敏。
- 限制工作目录。

面试中可以说：编码 Agent 最大风险不是“回答错”，而是“自动把错执行到真实仓库里”。

<h2 id="q-009">面试问题：如何处理远程会话或云端执行中的权限请求？</h2>

远程 Agent 通常在云端容器或远程 worker 中执行工具，而用户在本地 UI。此时权限请求需要跨会话桥接：

1. 远程执行器生成工具权限请求。
2. 本地 UI 收到请求并展示工具名、参数、风险说明。
3. 用户允许或拒绝。
4. 本地把 permission response 发回远程 session。
5. 远程继续执行或终止。

关键设计：

- 请求必须有唯一 ID。
- 权限请求可以被取消。
- 断线后要能恢复 pending 状态。
- 本地没有完整工具实现时，也要能展示工具 stub。
- 所有响应要进入审计日志。

---

<h1 id="q-010">3. 编码 Agent 如何理解一个真实代码仓库？</h1>

真实代码仓库比单个 prompt 复杂得多。编码 Agent 需要分层理解：

- 项目结构。
- 构建和测试命令。
- 代码风格。
- 模块边界。
- 当前用户改动。
- 历史问题和 TODO。
- 依赖和运行环境。
- 相关文件之间的关系。

常用手段：

- `rg --files` / `rg` 快速搜索。
- 读取 README、AGENTS.md、CLAUDE.md。
- 使用 LSP 获取诊断和引用。
- 使用 Git diff 识别当前改动。
- 读取测试失败日志。
- 维护文件读缓存和索引。

<h2 id="q-011">面试问题：CLAUDE.md / AGENTS.md 这类项目记忆文件有什么作用？</h2>

本节从代码仓库工程实践解释项目记忆文件；项目记忆作为上下文工程模式、与 Skill / Memory / RAG 的关系，见 [05_Agent记忆与上下文工程高频考点.md](05_Agent记忆与上下文工程高频考点.md)。

它们是给 Agent 看的项目说明书。

常见内容：

- 项目简介。
- 目录结构。
- 构建命令。
- 测试命令。
- 代码规范。
- 常见坑。
- 提交流程。
- 禁止操作。
- 重要设计约束。

价值：

- 减少每次都重新探索项目。
- 保持团队规范一致。
- 降低工具误用。
- 让 Agent 更快进入有效工作状态。

区别：

- `AGENTS.md` 更偏开放格式和多 Agent 通用约定。
- `CLAUDE.md` 是特定生态中的项目/用户记忆文件。

本质上，二者都属于项目级 Agent 指令与记忆入口：让 Agent 在进入仓库前知道“这里如何构建、如何测试、哪些边界不能碰、团队偏好是什么”。

<h2 id="q-012">面试问题：上下文压缩、摘要和恢复在编码 Agent 中如何工作？</h2>

这里聚焦编码 Agent 的 session compact、仓库状态恢复、diff 和测试日志压缩；通用上下文窗口治理和 Context Engine 机制，见 [05_Agent记忆与上下文工程高频考点.md](05_Agent记忆与上下文工程高频考点.md)。

长任务会产生大量工具输出、文件内容和对话历史。上下文压缩的目标是把历史浓缩成可继续工作的状态。

好的压缩摘要应包含：

- 用户目标。
- 当前计划。
- 已完成步骤。
- 修改过的文件。
- 重要代码位置。
- 已运行命令和结果。
- 失败尝试。
- 未解决问题。
- 下一步建议。
- 用户约束。

恢复流程：

1. 加载 session 历史或 compact summary。
2. 恢复当前工作目录、模型、工具、权限模式。
3. 检查文件系统状态和 Git diff。
4. 重新注入项目记忆。
5. 让 Agent 从摘要中的下一步继续。

<h2 id="q-013">面试问题：代码索引、ripgrep、LSP、文件缓存如何协同？</h2>

| 能力 | 优点 | 适合问题 |
| --- | --- | --- |
| ripgrep | 快速、简单、可靠 | 搜文本、定位调用、找配置 |
| 文件缓存 | 减少重复读取 | 已看过的文件和片段 |
| LSP | 语义级理解 | 定义、引用、诊断、类型 |
| 代码索引 | 跨仓库结构化检索 | 大型项目、符号和依赖图 |
| Git diff | 当前变更状态 | 避免覆盖用户改动 |

工程建议：

- 先用 `rg` 定位候选文件。
- 再读取关键片段。
- 对类型/引用问题使用 LSP。
- 对大仓库维护索引。
- 每次写入前看 diff 和文件状态。

<h2 id="q-014">面试问题：如何避免把无关文件塞满上下文？</h2>

策略：

- 先搜索后读取，避免盲目打开大量文件。
- 只读相关片段，而不是整文件。
- 对重复工具输出做折叠。
- 对日志只保留错误附近上下文。
- 用子 Agent 探索并返回摘要。
- 用结构化计划记录状态，而不是保留所有历史。
- 将低价值历史压缩成 summary。
- 根据任务阶段切换上下文：探索、实现、验证、总结。

---

<h1 id="q-015">4. 编码 Agent 为什么需要子 Agent 和 Skills？</h1>

复杂编码任务常常需要并行探索、分工实现和专业流程。子 Agent 和 Skills 分别解决两个问题：

- **子 Agent**：把任务拆给另一个模型会话，隔离上下文，支持并行。
- **Skills**：把可复用的专业知识、流程、模板和脚本打包，按需加载。

它们能降低主 Agent 上下文压力，并提升复杂任务的专业度。

<h2 id="q-016">面试问题：子 Agent 适合处理哪些任务？</h2>

适合：

- 大仓库并行探索。
- 独立模块实现。
- 测试修复。
- 代码审查。
- 文档补充。
- 问题复现。
- 多方案对比。

不适合：

- 需要立即阻塞主流程的下一步。
- 写同一文件的并行任务。
- 模糊且高度耦合的决策。
- 需要强全局一致性的核心设计。

设计要点：

- 子任务要具体、边界清晰。
- 明确可写文件范围。
- 返回变更文件和结论。
- 不要重复分派同一个问题。
- 主 Agent 负责整合和最终验证。

<h2 id="q-017">面试问题：Skills 的本质是什么？与工具有什么区别？</h2>

Skill 是按需加载的“能力包”。它通常包含：

- 触发条件。
- 操作流程。
- 专业知识。
- 模板。
- 脚本。
- 参考资料。
- 资产文件。

工具是“能执行的动作”，Skill 是“如何完成某类任务的方法论和资源包”。

对比：

| 维度 | Tool | Skill |
| --- | --- | --- |
| 核心 | 执行动作 | 提供流程和知识 |
| 形式 | 函数/API/命令 | SKILL.md + 脚本/模板 |
| 调用 | 模型直接调用 | 先加载说明再执行流程 |
| 例子 | read_file、bash、search | code-review、imagegen、openai-docs |

Skills 的价值是渐进式上下文：只有任务需要时才加载，避免系统 prompt 过胖。

<h2 id="q-018">面试问题：Hooks 在编码 Agent 中有什么价值？</h2>

Hooks 是在 Agent 生命周期关键点插入自定义逻辑。

常见类型：

- session_start：会话开始时加载上下文。
- pre_tool：工具调用前做安全检查。
- post_tool：工具调用后记录结果或更新状态。
- pre_compact：压缩前保存重要信息。
- post_compact：压缩后验证摘要。
- post_sampling：模型输出后触发文档更新或检查。

价值：

- 安全：阻止危险工具调用。
- 审计：记录所有关键动作。
- 自动化：任务结束后更新文档、生成报告。
- 质量：运行格式化、lint、测试。
- 记忆：提取长期偏好和项目知识。

<h2 id="q-019">面试问题：Plan Mode、Todo、任务状态如何提升长任务可靠性？</h2>

长任务失败常见原因是目标漂移和状态遗忘。Plan Mode 和 Todo 能把隐式思考变成显式状态。

作用：

- 让用户先确认方案。
- 降低误执行风险。
- 将大任务拆成可追踪步骤。
- 防止遗漏验证。
- 在上下文压缩后恢复进度。
- 让 UI 展示当前任务状态。

好的任务状态应包含：

- pending。
- in_progress。
- completed。
- blocked。

并且一次只应有一个核心 in_progress，避免 Agent 同时追多个主线导致混乱。

---

<h1 id="q-020">5. 编码 Agent 为什么常同时支持 CLI、IDE 和远程会话？</h1>

不同入口解决不同工作流：

- CLI：适合终端用户、脚本化、仓库级任务。
- IDE：适合编辑器上下文、选区、诊断、跳转、diff 展示。
- 远程会话：适合云端执行、移动端查看、长任务后台运行。

成熟编码 Agent 通常需要三者协同。

<h2 id="q-021">面试问题：CLI Agent 和 IDE Agent 的体验差异是什么？</h2>

| 维度 | CLI | IDE |
| --- | --- | --- |
| 优势 | 接近真实开发环境，命令强 | 有编辑器上下文和可视化 |
| 上下文 | cwd、Git、Shell、文件系统 | 打开文件、选区、诊断、项目树 |
| 展示 | 文本流、命令输出 | diff、inline edit、问题面板 |
| 用户 | 终端熟练开发者 | 日常 IDE 开发者 |
| 风险 | 命令副作用更强 | 编辑器状态同步复杂 |

<h2 id="q-022">面试问题：IDE Bridge 通常需要传递哪些信息？</h2>

IDE Bridge 需要双向通信：

**IDE 到 Agent：**

- 当前 workspace。
- 打开的文件。
- 光标位置和选区。
- 诊断信息。
- 用户点击的文件引用。
- IDE 配置和语言服务状态。

**Agent 到 IDE：**

- 打开文件。
- 展示 diff。
- 定位行号。
- 请求用户确认。
- 显示进度和通知。
- 应用编辑。

安全要点：

- Bridge 连接需要身份认证。
- 本地端口不应被任意网页调用。
- 远程控制要有 trusted device 或 token。
- 权限回调必须可审计。

<h2 id="q-023">面试问题：远程编码 Agent 如何处理 session、interrupt、resume 和 reconnect？</h2>

远程会话的关键是可靠性。

需要设计：

- session id 标识会话。
- WebSocket 或 streaming 接收事件。
- HTTP/控制通道发送用户消息和权限响应。
- interrupt 信号取消当前任务。
- reconnect 策略处理网络波动。
- session not found 的短暂重试。
- compact boundary 表示历史压缩点。
- resume 恢复摘要和状态。
- remote permission bridge 处理云端工具权限。

面试中可以强调：远程 Agent 不是简单把 CLI 放到服务器上，而是要处理流式事件、权限回调、断线恢复和状态一致性。

<h2 id="q-024">面试问题：MCP 在编码 Agent 中通常如何使用？</h2>

本节只说明 MCP 在编码场景中的工具接入方式；MCP 协议结构、传输、能力协商和安全设计，见 [04_MCP与A2A协议高频考点.md](04_MCP与A2A协议高频考点.md)。

编码 Agent 可通过 MCP 接入：

- GitHub/GitLab。
- Issue/PR 系统。
- 数据库。
- 文档平台。
- 浏览器自动化。
- 云服务。
- 内部代码搜索。
- 企业知识库。

优势：

- 工具连接器复用。
- 本地和远程工具统一协议。
- Server 可独立维护。
- 支持资源、提示模板和工具发现。

风险：

- MCP Server 本身可能暴露敏感能力。
- 工具返回内容可能包含 prompt injection。
- 远程 MCP 必须做认证和授权。
- 工具描述要清楚标注副作用。

---

<h1 id="q-025">6. 如何评估编码 Agent 是否真的有用？</h1>

本章的评测只围绕编码 Agent：issue 修复、测试、diff、代码质量、仓库权限和工程交付。通用 Agent Benchmark、Eval Harness、AgentOps 和事故响应，见 [06_Agent安全评测与AgentOps高频考点.md](06_Agent安全评测与AgentOps高频考点.md)。

编码 Agent 评估应该同时看任务成功、工程质量、安全和成本。

指标包括：

- Issue 修复成功率。
- 测试通过率。
- 编译/lint 通过率。
- diff 大小和侵入性。
- 是否覆盖边界 case。
- 是否引入回归。
- 人工 review 通过率。
- 平均任务时长。
- 工具调用次数。
- token 和费用。
- 权限请求次数。
- 失败恢复能力。

<h2 id="q-026">面试问题：SWE-bench 类评测为什么重要但不够？</h2>

SWE-bench 评估真实 GitHub issue 修复能力，非常重要，因为它接近真实软件工程。

但它不够覆盖：

- 企业私有仓库规范。
- 多语言复杂构建。
- 长期任务和跨 PR 协作。
- 人类交互和权限审批。
- 代码审查质量。
- 安全合规。
- 性能优化和架构设计。
- 文档、迁移、配置类任务。

因此生产评估还需要自建任务集，包括：

- 常见 bug。
- 真实历史 issue。
- 重构任务。
- 测试补齐。
- 文档生成。
- 安全修复。
- CI 失败修复。

<h2 id="q-027">面试问题：编码 Agent 的 tracing 和 telemetry 应该记录什么？</h2>

建议记录：

- 用户目标。
- 模型版本。
- 系统提示词版本。
- 工具列表。
- 每次工具调用和参数。
- 权限请求和结果。
- 文件变更 diff。
- 命令输出和退出码。
- 测试结果。
- 子 Agent 任务和结论。
- 上下文压缩边界。
- token、耗时、费用。
- 最终交付摘要。

隐私要求：

- 可关闭遥测。
- 敏感文件脱敏。
- 不上传密钥。
- 企业可配置日志保留周期。
- 用户可查看和删除历史。

<h2 id="q-028">面试问题：编码 Agent 常见失败模式有哪些？</h2>

| 失败模式 | 原因 | 缓解 |
| --- | --- | --- |
| 改错文件 | 搜索不充分，误解架构 | 先定位引用和测试，要求计划 |
| 覆盖用户改动 | 未检查 dirty state | 写前检查 diff |
| 测试没跑 | 任务结束过早 | Todo 中强制验证步骤 |
| 幻觉 API | 未读类型定义或文档 | LSP/rg/官方文档 |
| 命令危险 | 权限过宽 | 沙箱和审批 |
| 上下文污染 | 历史过长且无关 | 压缩和选择性保留 |
| 长任务漂移 | 缺少显式计划 | Plan Mode + TODO |
| 子 Agent 冲突 | 写同一文件 | 明确文件所有权 |
| Prompt injection | 工具返回恶意内容 | 内容隔离和策略检查 |

<h2 id="q-029">面试问题：如何设计企业内可用的编码 Agent？</h2>

建议路线：

1. **只读阶段**

   先做代码问答、搜索、解释、review，不允许写文件。

2. **建议阶段**

   生成 patch，但由开发者应用。

3. **受控写入阶段**

   允许修改分支内文件，但所有 diff 需审查。

4. **验证阶段**

   允许运行测试、lint、构建，但禁止危险命令。

5. **PR 阶段**

   自动创建 PR，必须人工 review 后合并。

6. **持续优化**

   用真实任务、失败 case、review 反馈优化工具、prompt 和策略。

企业必备能力：

- SSO/RBAC。
- 仓库权限继承。
- 审计日志。
- 网络隔离。
- secret scanning。
- 私有 MCP Server。
- 模型和数据策略。
- 评测集和灰度发布。

---

<h1 id="q-030">7. AgentOS / 个人助理型 Agent 系统通常如何分层？</h1>

本章使用 AgentOS 讨论“运行时工程层”，重点是入口、Gateway、Session、Runtime、Context Engine、Tool Policy 和后台任务。企业平台中的 Builder、Registry、Marketplace、多租户产品治理和商业选型，见 [07_企业级Agent平台与产品落地高频考点.md](07_企业级Agent平台与产品落地高频考点.md)。

AgentOS 可以理解为把 Agent 从“单次对话应用”升级为“长期运行的智能操作系统层”。它不只管理模型调用，还要管理消息入口、会话、工具、权限、上下文、任务、事件、自动化和多 Agent 隔离。

通用分层如下：

1. **交互入口层**

   包括 CLI、IDE、Web、移动端、桌面端、聊天软件、语音入口、Webhook、定时任务等。它解决“用户和外部事件从哪里进入 Agent”的问题。

2. **Gateway 控制平面**

   负责统一接入消息、维护连接、路由会话、调度运行时、转发事件、管理认证、暴露状态和处理重连。

3. **Session 与 Channel 层**

   管理不同来源的上下文边界，例如用户私聊、群聊、项目空间、定时任务、浏览器会话、远程工作区。

4. **Agent Runtime 层**

   负责模型循环、工具调用、上下文拼装、streaming、重试、停止条件和 turn 生命周期。

5. **Context Engine 层**

   负责记忆、项目知识、历史摘要、token 预算、上下文压缩和检索增强。

6. **Tool / Skill / Plugin 层**

   提供文件系统、Shell、浏览器、代码工具、消息发送、日历、任务系统、媒体生成、企业 API、MCP Server 等能力。

7. **安全与治理层**

   包括沙箱、工具策略、权限审批、密钥隔离、审计日志、敏感内容脱敏、数据保留策略。

8. **任务与自动化层**

   支持后台任务、cron、hook、workflow、异步通知、断线恢复和长任务状态追踪。

面试中可以总结为：AgentOS 的核心不是“多接几个工具”，而是把 Agent 的入口、运行时、上下文、安全和任务生命周期系统化。

<h2 id="q-031">面试问题：Gateway 控制平面在 Agent 系统中负责什么？</h2>

Gateway 是长期运行 Agent 系统里的控制平面。它通常不直接负责模型推理细节，而是负责把各种外部入口、运行时、工具节点和事件流组织起来。

典型职责包括：

- 管理 WebSocket、HTTP、CLI、IDE、移动端、聊天软件等连接。
- 接收用户消息、系统事件、定时任务、Webhook。
- 根据 channel、account、peer、workspace、agent id 路由到正确 session。
- 管理 session 的 start、resume、interrupt、reset、compact、stop。
- 转发模型回复、工具事件、进度事件、权限请求和后台任务状态。
- 维护在线状态、心跳、健康检查和重连。
- 管理设备配对、token、可信代理、本地网络或内网连接。
- 提供统一协议 Schema，避免不同客户端各说各话。
- 对副作用操作支持 idempotency key，防止断线重试造成重复执行。

为什么需要 Gateway？

- 多端入口如果直接连运行时，权限、状态和上下文容易碎片化。
- 长任务需要持续事件流和断线恢复。
- 多 Agent、多账号、多 channel 需要统一路由。
- 企业场景需要集中审计、策略和可观测性。

一个成熟设计通常会把 Gateway 当作“事件总线 + 会话路由器 + 控制面 API”，把具体模型循环交给 Agent Runtime。

<h2 id="q-032">面试问题：Provider、Model、Agent Runtime、Channel 如何解耦？</h2>

这四个概念经常被混在一起，但工程上必须拆开：

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| Provider | 模型或 Agent 能力的供应方 | OpenAI、Anthropic、本地模型服务、企业私有网关 |
| Model | 具体推理模型 | GPT、Claude、Gemini、Qwen、Llama 等具体版本 |
| Agent Runtime | 负责模型循环和工具执行的运行时 | 编码 Agent runtime、浏览器 Agent runtime、多模态 Agent runtime |
| Channel | 用户或事件进入 Agent 的通道 | CLI、IDE、Slack、Discord、Web、Webhook、Cron |

解耦的好处：

- 同一个 channel 可以切换不同 runtime。
- 同一个 runtime 可以接多个 provider/model。
- 同一个模型可以服务编码、浏览器、客服、数据分析等不同 Agent。
- 企业可以通过配置控制模型供应商，而不改业务入口。
- 运行时可以独立处理上下文压缩、工具协议、权限和 tracing。

运行时选择通常需要回答几个问题：

- 谁拥有模型 loop？
- 谁保存 canonical thread history？
- 工具调用是模型原生支持，还是运行时模拟？
- 动态工具、MCP、hooks、subagent 是否可用？
- 上下文压缩和 session 恢复由谁负责？
- 当指定 runtime 不可用时，是降级、排队还是失败？

面试中可以说：Provider / Model 解决“用什么脑”，Runtime 解决“怎么思考和行动”，Channel 解决“从哪里接活”。

<h2 id="q-033">面试问题：多通道 Agent 如何做会话路由和用户隔离？</h2>

多通道 Agent 会同时面对私聊、群聊、工作区、Webhook、定时任务、IDE、CLI 等来源。核心问题是：同一条消息应该进入哪个 session，以及哪些上下文绝不能互相看到。

常见路由维度：

- channel：Slack、Discord、Telegram、Web、CLI、IDE、Email。
- account：同一平台上的不同账号或 workspace。
- peer：私聊对象、群、频道、组织。
- agent：不同人格、职责或工作区的 Agent。
- source type：用户消息、系统事件、cron、webhook、后台任务回调。

常见 session scope：

| Scope | 适用场景 | 风险 |
| --- | --- | --- |
| main session | 单用户本地 Agent | 简单但不适合多人 |
| per-peer session | 每个私聊对象独立上下文 | 能避免用户间上下文泄露 |
| per-channel-peer session | 同一平台不同群/频道隔离 | 适合团队协作 |
| per-account-channel-peer session | 多账号、多组织、多平台 | 最安全但管理复杂 |

关键原则：

- 私聊默认必须隔离，不能让 Alice 的上下文出现在 Bob 的会话中。
- 群聊与私聊应分离，群聊中的上下文不能自动进入私人会话。
- 自动任务和 webhook 应有独立上下文或明确绑定上下文。
- 路由规则要可解释，冲突时采用“最具体规则优先”。
- session reset、daily reset、idle reset、manual reset 要有明确策略。
- transcript、memory、tool trace 要跟 session id 强绑定。

面试回答时可以强调：多通道 Agent 的安全事故往往不是模型能力不足，而是会话边界设计错误导致上下文串线。

<h2 id="q-034">面试问题：多 Agent 系统如何做到 workspace、auth、session 隔离？</h2>

多 Agent 系统中，“一个 Agent”不只是一个 prompt，而是一组隔离资源：

- persona / system prompt。
- workspace / cwd。
- agent directory / state。
- session store。
- memory store。
- auth profile。
- tool policy。
- sandbox policy。
- channel binding。

推荐设计：

1. **Workspace 隔离**

   不同 Agent 默认使用不同工作区。即使共享同一仓库，也要明确只读/读写边界和文件所有权。

2. **Auth 隔离**

   不同 Agent 使用不同 token、cookie、MCP credential、云账号或最小权限角色。不要让客服 Agent 复用工程 Agent 的代码仓库权限。

3. **Session 隔离**

   每个 Agent 维护独立会话历史和压缩摘要，避免人格、任务和上下文混杂。

4. **Tool Policy 隔离**

   不同 Agent 可用工具不同。例如代码 Agent 可以读写仓库，客服 Agent 只能查知识库和创建工单。

5. **Memory 隔离**

   长期记忆默认分域保存。跨 Agent 检索必须做权限过滤、脱敏和来源标注。

6. **Channel Binding**

   将特定 channel、account、peer 绑定到特定 Agent，并支持最具体规则优先。

一个常见误区是只给不同 Agent 换 system prompt，却复用同一工作目录、会话存储和凭证。这样看起来是多 Agent，实际上只是多个名字共用同一个安全边界。

---

<h1 id="q-035">8. AgentOS 中 Sandbox、Tool Policy、Elevated 为什么要分开？</h1>

Sandbox、Tool Policy、Elevated 是三类不同控制，不能混为一谈：

| 控制项 | 决定什么 | 例子 |
| --- | --- | --- |
| Sandbox | 工具在哪里运行、能访问哪些系统资源 | 本机、容器、远程主机、只读工作区 |
| Tool Policy | 哪些工具可用、哪些参数允许、是否需要审批 | 禁止发消息、允许读文件、写文件需确认 |
| Elevated | 某些执行类工具是否能临时跳出沙箱或提升执行权限 | 在用户确认后运行需要宿主机权限的命令 |

为什么要分开？

- 允许工具不等于允许访问宿主机。
- 在沙箱内运行命令不等于这个命令安全。
- 提权执行不应该自动获得更多工具能力。
- 企业策略需要独立控制“工具集合”和“执行环境”。

典型规则：

- deny 优先级最高。
- allow 列表非空时，未列出的工具默认不可用。
- 高风险工具即使在沙箱中也可能需要审批。
- Elevated 只影响执行位置或权限，不改变工具授权。
- Docker socket、宿主机挂载、共享密钥会穿透沙箱，需要单独禁用或强审计。

面试中可以举例：一个 Agent 可以被允许使用 `bash`，但只能在容器内执行；另一个 Agent 可以访问消息工具，但不能发送外部消息，只能草拟。两者分别由 Sandbox 和 Tool Policy 控制。

<h2 id="q-036">面试问题：Context Engine 的 ingest / assemble / compact / after-turn 生命周期如何理解？</h2>

Context Engine 是 AgentOS 中负责“上下文生命周期”的模块。它决定模型每一轮看到什么、历史如何压缩、长期记忆如何更新。

典型生命周期：

1. **ingest**

   摄取新信息，例如用户消息、工具结果、文件变化、网页内容、语音转写、外部事件。

2. **assemble**

   为当前 turn 拼装上下文，包括 system prompt、开发者指令、项目规则、session 摘要、相关记忆、检索片段、打开文件、当前任务状态。

3. **compact**

   当上下文接近 token 预算时，把历史压缩成结构化摘要，保留目标、约束、已完成动作、重要文件、失败尝试、下一步。

4. **after-turn**

   模型回复和工具调用结束后，更新任务状态、写入记忆、保存 trace、触发后处理 hook。

好的 Context Engine 要处理：

- prompt authority：系统指令、项目规则、用户消息、工具输出的优先级不能混乱。
- token budget：核心目标、当前文件、错误日志优先，低价值历史可压缩。
- provenance：记忆和检索片段要保留来源，避免把不可信内容当系统指令。
- phase awareness：探索、实现、验证、总结阶段需要不同上下文。
- subagent context：子 Agent 只拿必要上下文，结束后返回摘要而不是倾倒全部历史。

一句话：Context Engine 是 Agent 的“工作记忆调度器”，比简单 RAG 更关注 turn 生命周期和指令优先级。

<h2 id="q-037">面试问题：Hooks、Cron/Webhook、Background Task 在 AgentOS 中分别解决什么问题？</h2>

三者都和自动化有关，但职责不同：

| 机制 | 核心作用 | 典型场景 |
| --- | --- | --- |
| Hooks | 在 Agent 生命周期关键点插入逻辑 | 消息预处理、工具调用前审批、压缩前保存、会话结束归档 |
| Cron/Webhook | 从时间或外部系统触发 Agent | 每日总结、CI 失败通知、工单事件、监控告警 |
| Background Task | 追踪异步长任务状态 | 长时间构建、代码生成、浏览器任务、媒体处理、远程子任务 |

Hooks 常见触发点：

- message received / preprocessed / sent。
- session start / end。
- before prompt build。
- before model call。
- before / after tool call。
- before / after compaction。
- agent end。
- gateway startup / shutdown。

Background Task 常见状态：

$$
\text{queued} \rightarrow \text{running} \rightarrow \text{succeeded / failed / timed\_out / cancelled / lost}
$$

设计要点：

- Hooks 必须可观测，失败策略要明确，是阻断、跳过还是降级。
- Cron/Webhook 触发的任务要绑定 session 或创建独立 session。
- Background Task 是活动账本，不一定是调度器本身。
- 任务完成通知要支持直接推送，也要支持离线后排队等待用户回来。
- 自动化能力要受 Tool Policy 和权限模型约束。

面试中可以说：Hooks 解决“生命周期扩展”，Cron/Webhook 解决“外部触发”，Background Task 解决“异步状态追踪”。

<h2 id="q-038">面试问题：如何设计支持长任务、断线恢复和事件流的 Agent 运行时？</h2>

长任务 Agent 的关键难点是：模型会分多轮调用工具，工具可能长时间运行，用户可能中途断线或打断，系统还要能恢复上下文。

推荐设计：

1. **单 session 串行运行**

   同一 session 同一时间只允许一个主 run 写入状态，避免并发工具调用把 transcript 和文件状态写乱。

2. **事件流协议**

   将 assistant token、tool start、tool progress、tool result、permission request、task state、error、final answer 都作为结构化事件发送给客户端。

3. **可恢复 session store**

   持久化 messages、compact summary、tool trace、pending permission、background task、当前计划和工作目录。

4. **Interrupt / Resume**

   用户可以中断运行，运行时需要安全停止当前工具或标记后台任务，并在恢复时重新加载状态。

5. **幂等工具调用**

   对发消息、创建工单、写数据库、提交 PR 等副作用操作使用 idempotency key，避免重试导致重复执行。

6. **权限请求状态机**

   权限请求必须有唯一 ID、过期时间、取消状态和审计记录。断线重连后客户端仍能看到 pending request。

7. **上下文压缩边界**

   长任务中定期 compact，把旧工具输出压缩成可继续工作的摘要，避免 token 爆炸和任务漂移。

8. **任务通知策略**

   对后台任务支持 done only、state changes、silent 等通知策略，避免用户被刷屏。

9. **超时和孤儿任务处理**

   对长时间无心跳的任务标记 lost 或 timed out，并提供人工恢复入口。

这类运行时的成熟度，决定了 Agent 能否从“演示型聊天机器人”变成“可以接真实工作流的工程系统”。

---

## 高频速记

1. 编码 Agent 是能读仓库、改文件、跑命令、验证结果的软件工程执行器。
2. 工具系统必须包含 Schema、权限、执行、进度、错误和审计。
3. Bash 和写文件工具必须强权限控制。
4. 项目记忆文件让 Agent 理解仓库规则和构建测试方式。
5. 上下文压缩要保留目标、进度、变更、失败尝试和下一步。
6. 子 Agent 适合并行探索和独立模块任务，不适合抢同一文件。
7. Skills 是按需加载的方法论和资源包，工具是可执行动作。
8. IDE Bridge 提供选区、诊断、diff 和可视化交互。
9. 远程 Agent 要处理 session、权限回调、interrupt、resume 和 reconnect。
10. 企业编码 Agent 应从只读、建议、受控写入逐步上线。
11. AgentOS 要统一入口、Gateway、Session、Runtime、Context Engine、工具、安全和后台任务。
12. Gateway 是控制平面，负责连接、路由、事件流、认证、状态和断线恢复。
13. Provider、Model、Runtime、Channel 要解耦，避免把模型供应商和产品入口绑死。
14. 多通道 Agent 的核心风险是 session 串线，私聊、群聊、Webhook、Cron 必须明确隔离。
15. Sandbox 控制执行环境，Tool Policy 控制工具授权，Elevated 控制临时提权，三者不能混用。
16. Context Engine 管理 ingest、assemble、compact、after-turn，是比简单 RAG 更完整的上下文生命周期系统。
17. Hooks 解决生命周期扩展，Cron/Webhook 解决外部触发，Background Task 解决异步状态追踪。

## 参考资料

- 本地研究资料：`/Users/rocky/Desktop/AIGC技术知识/研究项目/claude-code`，用于提炼通用编码 Agent 工程模式。
- 本地研究资料：`/Users/rocky/Desktop/AIGC技术知识/研究项目/Claude Code泄漏源码干货资源`，仅作为架构研究素材使用，本文不复刻具体实现细节。
- 本地研究资料：`/Users/rocky/Desktop/AIGC技术知识/AI Agent/openclaw`，用于提炼通用 AgentOS、Gateway、Session、Runtime、Sandbox、Hooks 与后台任务设计。
- Anthropic, [**Claude Code Documentation**](https://docs.anthropic.com/en/docs/claude-code/overview).
- agentsmd, [**AGENTS.md: A Simple Open Format for Guiding Coding Agents**](https://github.com/agentsmd/agents.md).
- OpenAI, [**Codex Documentation**](https://developers.openai.com/codex/).
- Jimenez et al., [**SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**](https://www.swebench.com/), 2024.
- OpenAI, [**Agents SDK Documentation**](https://openai.github.io/openai-agents-python/).
- Model Context Protocol, [**Specification**](https://modelcontextprotocol.io/specification/2025-06-18).
- A2A Project, [**Agent2Agent Protocol Specification**](https://a2aproject.github.io/A2A/latest/specification/).
- LangChain, [**LangGraph Documentation**](https://docs.langchain.com/oss/python/langgraph/overview).
