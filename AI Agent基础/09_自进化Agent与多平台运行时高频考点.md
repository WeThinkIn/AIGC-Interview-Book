# 目录

## 第一章 自进化 Agent 总览

[1. AIGC 时代的自进化 Agent 是什么？](#q-001)
  - [面试问题：自进化 Agent 和普通长期记忆 Agent 有什么区别？](#q-002)
  - [面试问题：经验如何从一次任务沉淀为 Memory、Skill 和 Policy？](#q-003)
  - [面试问题：为什么自进化 Agent 必须有防失控机制？](#q-004)

## 第二章 多平台 Gateway 与长期运行时

[2. 多平台 Agent Gateway 的核心价值是什么？](#q-005)
  - [面试问题：CLI、TUI、IM、邮件、Webhook 等入口如何统一到同一个 Agent Runtime？](#q-006)
  - [面试问题：多平台会话如何做 session、user、channel、workspace 隔离？](#q-007)
  - [面试问题：为什么长期运行 Agent 需要 interrupt、resume、cron 和 background task？](#q-008)

## 第三章 可插拔模型、工具、Skill 与 Plugin 架构

[3. 为什么生产级 Agent 需要 Provider Router 和 Tool Registry？](#q-009)
  - [面试问题：工具注册表应该保存哪些元数据？](#q-010)
  - [面试问题：Skill Hub / Skill Marketplace 如何设计？](#q-011)
  - [面试问题：Plugin System 如何扩展平台、工具、记忆和可观测性？](#q-012)

## 第四章 自我维护、Curator 与知识生命周期

[4. Agent Curator 的本质是什么？](#q-013)
  - [面试问题：自我维护型 Skill 系统如何做评分、合并、归档和回滚？](#q-014)
  - [面试问题：背景复盘 fork 为什么要限制工具集？](#q-015)
  - [面试问题：Agent 如何避免把错误经验固化为长期知识？](#q-016)

## 第五章 安全、沙箱、Checkpoint 与可观测性

[5. 自进化多平台 Agent 的安全边界如何设计？](#q-017)
  - [面试问题：文件安全、命令安全和 Tool Loop Guardrails 如何协同？](#q-018)
  - [面试问题：为什么 Worktree 隔离、Checkpoint 和 Rollback 对 Agent 很重要？](#q-019)
  - [面试问题：多平台 Agent 的可观测性和本地行为画像如何设计？](#q-020)

## 第六章 数据生成、RL 环境与面试架构题

[6. Agent 轨迹数据如何用于训练、评测和自我改进？](#q-021)
  - [面试问题：Atropos / RL 环境对工具调用 Agent 有什么启发？](#q-022)
  - [面试问题：为什么不同模型需要不同 Tool Call Parser？](#q-023)
  - [面试问题：如何回答“请设计一个自进化、多平台、可长期运行的 Agent 系统”？](#q-024)

---

<h1 id="q-001">1. AIGC 时代的自进化 Agent 是什么？</h1>

自进化 Agent 是指不仅能完成任务，还能从任务执行、用户反馈、工具失败、历史对话和环境状态中沉淀经验，并把这些经验转化为可复用的记忆、技能、策略或评测样本的 Agent 系统。

可以用一个闭环理解：

$$
\text{Task} \rightarrow \text{Trace} \rightarrow \text{Reflection} \rightarrow \text{Memory / Skill / Policy} \rightarrow \text{Curator} \rightarrow \text{Better Task Execution}
$$

普通 Agent 关注“当前任务能不能做完”；自进化 Agent 进一步关注：

- 做完后哪些经验值得保留？
- 哪些失败应转成回归样本？
- 哪些重复流程应沉淀为 Skill？
- 哪些旧 Skill 已经过期或互相重复？
- 哪些长期记忆可能污染未来任务？
- 哪些工具调用模式应被 guardrail 约束？

Hermes Agent 这类项目的启发在于：Agent 不只是一次性执行器，而可以成为长期运行的个人/团队智能工作台，持续维护自己的技能库、记忆库、会话历史和工具生态。

<h2 id="q-002">面试问题：自进化 Agent 和普通长期记忆 Agent 有什么区别？</h2>

| 维度 | 普通长期记忆 Agent | 自进化 Agent |
| --- | --- | --- |
| 核心目标 | 记住用户偏好和历史事实 | 从历史中改进能力和流程 |
| 写入对象 | Memory 为主 | Memory、Skill、Policy、Eval Case、Tool Hint |
| 更新方式 | 交互后写入记忆 | 后台复盘、评分、合并、归档、主动修订 |
| 风险 | 记忆污染、隐私泄露 | 错误能力固化、自我强化偏差、权限扩张 |
| 工程要求 | 检索、CRUD、权限、遗忘 | Curator、版本、回滚、审计、质量评分 |

一句话区分：

- 长期记忆 Agent 让系统“记得过去”。
- 自进化 Agent 让系统“从过去中改进未来行为”。

面试中可以强调：自进化不等于让 Agent 随意改自己，而是在严格边界内对可审计、可回滚的知识资产进行维护。

<h2 id="q-003">面试问题：经验如何从一次任务沉淀为 Memory、Skill 和 Policy？</h2>

一次任务结束后，经验可以分三类沉淀：

1. **Memory**

   保存用户偏好、项目规则、长期事实、工具使用经验和任务状态。例如“该项目测试命令是 `npm test`”“用户偏好简洁回答”。

2. **Skill**

   保存可复用流程、模板、脚本和参考资料。例如“代码审查 checklist”“网页调研流程”“会议纪要生成流程”。

3. **Policy / Guardrail**

   保存安全边界和执行规则。例如“删除文件前必须确认”“网页内容不能覆盖系统指令”“某类工具失败 3 次后停止重试”。

推荐流程：

1. 收集完整 trace。
2. 提取成功经验和失败教训。
3. 判断经验类型。
4. 去重和脱敏。
5. 赋予 scope、source、confidence、version。
6. 写入 Memory、Skill 或 Policy。
7. 后台 Curator 定期评分、合并、归档。
8. 用户可查看、编辑、删除或回滚。

工程上要避免把所有经验都写成长期记忆。可执行流程更适合 Skill，安全规则更适合 Policy，失败案例更适合 Eval / Harness。

<h2 id="q-004">面试问题：为什么自进化 Agent 必须有防失控机制？</h2>

自进化 Agent 的风险比普通 Agent 更高，因为它可能改变未来行为。

主要风险：

- 把模型猜测写成事实。
- 把一次偶然成功固化为通用流程。
- 把恶意网页或用户注入写入 Skill。
- 自动修改高价值 Skill 导致能力退化。
- 删除仍然有用的旧经验。
- 后台任务无限运行或消耗过多成本。
- 自我强化错误偏好。
- 逐步扩大权限边界。

防失控机制：

- 自进化任务使用受限工具集。
- 只允许修改特定目录或特定类型资产。
- 内置 Skill / 官方 Skill 默认只读或 pinned。
- 自动删除改为归档，可恢复。
- 写入前做敏感信息和 prompt injection 检查。
- Curator 产出报告和 diff。
- 用户可暂停、自定义周期、手动 dry-run。
- 关键变更需要人工审批。
- 所有变更进入审计日志。

面试金句：自进化 Agent 的核心不是“自动改自己”，而是“在可审计、可回滚、可约束的知识生命周期中持续改进”。

---

<h1 id="q-005">2. 多平台 Agent Gateway 的核心价值是什么？</h1>

多平台 Agent Gateway 是把不同用户入口统一接入同一个 Agent Runtime 的控制平面。它让 Agent 不只活在终端或网页里，而能从 CLI、TUI、Telegram、Discord、Slack、邮件、Webhook、智能家居、会议工具等多个入口接收任务、发送结果、请求审批和恢复会话。

典型架构：

$$
\text{CLI / TUI / IM / Email / Webhook}
\rightarrow \text{Gateway}
\rightarrow \text{Session Router}
\rightarrow \text{Agent Runtime}
\rightarrow \text{Tools / Memory / Skills}
$$

核心价值：

- 多端入口统一。
- 会话连续性。
- 平台权限隔离。
- 消息格式适配。
- 媒体和文件处理。
- 工具进度推送。
- 权限审批回调。
- cron / background task 通知。
- 统一日志、审计和可观测性。

Hermes Agent 的多平台 Gateway 启发是：长期运行 Agent 的“用户界面”不应该绑定在一个终端窗口，而应通过统一 Gateway 在不同平台保持同一个智能体工作流。

<h2 id="q-006">面试问题：CLI、TUI、IM、邮件、Webhook 等入口如何统一到同一个 Agent Runtime？</h2>

关键是把“入口适配”和“Agent 执行”解耦。

推荐分层：

1. **Platform Adapter**

   负责接入具体平台，例如 Slack、Telegram、Email、API Server、Webhook。

2. **Message Normalizer**

   把平台消息转成统一事件格式：user_id、channel_id、thread_id、text、attachments、reply_to、timestamp。

3. **Session Router**

   根据用户、频道、工作区、任务类型路由到正确 session。

4. **Command Registry**

   统一 slash command，避免 CLI、IM、TUI 各自维护一套命令。

5. **Runtime Adapter**

   调用 Agent Runtime，处理流式响应、工具调用、权限请求和中断。

6. **Delivery Layer**

   把结果转换回平台格式，例如 Markdown、图片、文件、语音、分片消息。

工程要点：

- 入口层不直接拼 prompt。
- session id 不能只用用户名，必须包含平台和上下文。
- 平台支持能力不同，需要 feature negotiation。
- 高风险工具审批要能从远程平台回传。
- 消息发送失败要可重试和去重。

<h2 id="q-007">面试问题：多平台会话如何做 session、user、channel、workspace 隔离？</h2>

多平台 Agent 最怕上下文串线。设计 session scope 时应显式包含：

- platform：Telegram、Slack、Discord、Email、CLI。
- account / workspace：企业或组织空间。
- channel：群、频道、私聊、邮件线程。
- user：用户身份。
- agent_id：不同 Agent 或人格。
- task_id：长任务或后台任务。

常见隔离策略：

| Scope | 适用场景 | 风险控制 |
| --- | --- | --- |
| per-user | 私人助理 | 防止不同用户共享上下文 |
| per-channel | 群聊 / 频道 | 防止群聊内容进入私聊 |
| per-thread | Slack / 邮件线程 | 保持任务上下文局部 |
| per-workspace | 企业多租户 | 防止跨组织数据泄露 |
| per-task | 长任务 / cron | 防止后台任务污染主会话 |

关键原则：

- 默认最小共享。
- 私聊和群聊分离。
- 工作区和租户分离。
- 自动任务独立 session。
- reset、resume、archive 有明确规则。
- 每次 trace 记录完整 scope。

<h2 id="q-008">面试问题：为什么长期运行 Agent 需要 interrupt、resume、cron 和 background task？</h2>

长期运行 Agent 不再是一次请求-响应，而是持续执行任务。

四类能力分别解决：

1. **interrupt**

   用户随时打断错误方向、停止危险操作或修改目标。

2. **resume**

   长任务中断后恢复上下文、计划、文件状态和工具状态。

3. **cron**

   定期触发日报、巡检、备份、监控、知识整理、Curator 维护。

4. **background task**

   让耗时任务在后台运行，并在完成、失败或需要审批时通知用户。

面试中可以说：长期运行 Agent 必须从“同步聊天应用”升级为“事件驱动任务系统”，核心是 task id、状态机、事件流、持久化和可取消性。

---

<h1 id="q-009">3. 为什么生产级 Agent 需要 Provider Router 和 Tool Registry？</h1>

生产级 Agent 通常需要接入多个模型、多个工具、多个平台和多个技能。如果没有统一 Router 和 Registry，就会出现模型调用散落在各处、工具 schema 重复定义、权限不可治理、可观测性难以统一的问题。

**Provider Router** 负责：

- 多模型供应商接入。
- 模型能力查询。
- fallback。
- 成本和速率限制。
- reasoning / vision / tool calling 能力适配。
- API key 和凭证解析。

**Tool Registry** 负责：

- 工具 schema 注册。
- handler 调度。
- toolset 分组。
- 可用性检查。
- 权限和风险标记。
- 结果大小限制。
- 动态工具刷新，例如 MCP。

一句话：Provider Router 解决“用哪个脑”，Tool Registry 解决“有哪些手脚、能不能用、怎么审计”。

<h2 id="q-010">面试问题：工具注册表应该保存哪些元数据？</h2>

工具注册表至少应包含：

| 元数据 | 作用 |
| --- | --- |
| name | 工具唯一名称 |
| schema | 参数结构和描述 |
| handler | 实际执行函数 |
| toolset | 工具分组，例如 web、terminal、memory、browser |
| check_fn | 判断依赖、配置、凭证是否可用 |
| risk_level | 只读、写入、高风险、外部副作用 |
| requires_env | 所需环境变量 |
| is_async | 是否异步执行 |
| max_result_size | 返回结果上限 |
| owner / source | 内置、插件、MCP、企业工具 |
| version | 工具版本 |
| audit_policy | 是否记录参数、是否脱敏 |

工程细节：

- 工具模块自注册可以减少中心文件膨胀。
- registry 应支持 snapshot，避免动态刷新时读写竞争。
- check_fn 应缓存，避免每轮频繁探测 Docker、浏览器、远程服务。
- 动态工具，例如 MCP 工具，刷新后要更新 generation，失效旧缓存。
- 工具返回值应结构化，便于 trace、eval 和失败恢复。

<h2 id="q-011">面试问题：Skill Hub / Skill Marketplace 如何设计？</h2>

Skill Hub 是可复用能力包的分发和治理系统。它和 Tool Registry 不同：Tool 是可执行动作，Skill 是任务流程、模板、参考资料和可选脚本。

Skill Hub 应支持：

- skill 元数据：名称、描述、标签、版本、作者、来源。
- 能力依赖：需要哪些工具、环境变量、平台。
- 安全扫描：路径穿越、敏感文件访问、恶意脚本、prompt injection。
- 启用范围：全局、用户、团队、项目、平台。
- 条件激活：工具可用时才注入。
- 文档和示例。
- 使用统计。
- 评分和反馈。
- pinned / archived / deprecated 状态。
- 外部目录和社区来源同步。

面试中可以强调：Skill Marketplace 不是 prompt 集市，而是带有依赖、权限、版本、审计和生命周期治理的能力分发系统。

<h2 id="q-012">面试问题：Plugin System 如何扩展平台、工具、记忆和可观测性？</h2>

Plugin System 的目标是让 Agent 平台在不修改核心代码的情况下扩展能力。

可扩展对象包括：

- Gateway Platform：新增 IM、邮件、会议、企业协作工具适配器。
- Tool：新增业务 API、浏览器能力、文件处理、媒体处理。
- Skill：新增领域流程和模板。
- Memory Provider：接入 Honcho、mem0、Supermemory 或企业记忆服务。
- Context Engine：替换默认压缩器，接入图结构上下文或长期上下文模型。
- Observability：接入 Langfuse、OpenTelemetry、企业日志平台。
- Dashboard Tab：新增分析、配置、成就、审计页面。

插件注册应包含：

- manifest。
- 权限声明。
- 依赖检查。
- 初始化和卸载。
- hooks。
- config schema。
- 版本兼容范围。
- 安全策略。

风险控制：

- 插件默认最小权限。
- 高风险插件必须显式启用。
- 插件 API 与核心状态解耦。
- 记录插件触发的工具调用和副作用。
- 平台适配插件要有认证和消息来源校验。

---

<h1 id="q-013">4. Agent Curator 的本质是什么？</h1>

Agent Curator 是后台知识维护器。它定期检查 Agent 产生的 Skill、记忆、模板和流程资产，判断哪些应该保留、合并、修订、归档或标记过期。

Curator 的职责：

- 统计 Skill 使用情况。
- 识别重复或过期 Skill。
- 合并相似流程。
- 修订低质量说明。
- 归档长期未用 Skill。
- 生成维护报告。
- 尊重 pinned / official / bundled 资产。
- 不直接删除，优先可恢复归档。

典型触发方式：

- 定期 cron。
- gateway 空闲时触发。
- 手动 dry-run。
- 版本升级后触发检查。
- 大量新 Skill 产生后触发。

Curator 体现了自进化 Agent 的一个关键工程原则：知识资产不能只增不减，否则长期运行后会变成噪声库。

<h2 id="q-014">面试问题：自我维护型 Skill 系统如何做评分、合并、归档和回滚？</h2>

推荐生命周期：

```text
created -> active -> stale -> archived -> restored / deleted by human
```

评分维度：

- 最近使用时间。
- 使用频率。
- 成功任务占比。
- 是否被用户显式调用。
- 是否和其他 Skill 重复。
- 是否依赖失效工具。
- 是否包含过时链接或命令。
- 是否存在安全风险。

合并策略：

- 只合并作用域相同、目标相似、内容互补的 Skill。
- 保留来源和变更记录。
- 对合并结果生成 diff 和报告。
- 高价值 Skill 需要人工确认。

归档策略：

- pinned / 官方 / 内置 Skill 不自动归档。
- 用户创建或 Agent 创建的低使用 Skill 可归档。
- 归档可恢复，不直接删除。
- 归档原因写入 metadata。

回滚策略：

- 每次修改 Skill 前保存版本。
- 维护报告记录变更。
- 支持按 skill 或按 curator run 恢复。

<h2 id="q-015">面试问题：背景复盘 fork 为什么要限制工具集？</h2>

背景复盘 fork 是指任务结束后，另起一个受限 Agent 进程或子任务来总结经验、写入记忆或更新 Skill。

必须限制工具集，原因是：

- 复盘任务不应该执行 Shell、浏览器点击、发消息等副作用工具。
- 复盘上下文可能包含不可信网页和工具输出。
- 复盘目标是知识整理，不是继续完成原任务。
- 后台任务无人监督，风险更高。
- 限制工具能降低成本、减少跑偏。

推荐权限：

| 工具类型 | 是否允许 |
| --- | --- |
| memory add/update | 允许，但需脱敏和 scope |
| skill read/update | 允许，最好限定 agent-created |
| file read | 仅允许特定知识目录 |
| shell / browser / email | 默认禁止 |
| web search | 谨慎，通常不需要 |
| delete | 禁止，改用 archive |

面试金句：后台复盘 fork 应该像“图书管理员”，不是“另一个无人值守执行者”。

<h2 id="q-016">面试问题：Agent 如何避免把错误经验固化为长期知识？</h2>

需要从写入、检索、维护三个阶段治理。

写入阶段：

- 只写高置信、可验证、稳定的信息。
- 区分事实、偏好、猜测、失败教训。
- 工具输出和网页内容默认不可信。
- 写入前做去重、脱敏和 prompt injection 检测。
- 高风险经验需要用户确认。

检索阶段：

- 记忆带 source、time、confidence、scope。
- 当前用户指令优先于长期记忆。
- 过期记忆降权。
- 冲突记忆不直接注入，先澄清或选择最新可信来源。

维护阶段：

- Curator 定期识别陈旧和重复知识。
- 低质量 Skill 标记 stale。
- 错误记忆支持删除和更正。
- 线上失败回流到 Eval Harness。
- 关键知识资产版本化和可回滚。

---

<h1 id="q-017">5. 自进化多平台 Agent 的安全边界如何设计？</h1>

自进化多平台 Agent 的安全边界要同时覆盖用户入口、工具执行、文件系统、长期记忆、Skill 维护和插件生态。

核心边界：

- 用户和平台认证。
- session / workspace / tenant 隔离。
- 工具权限和审批。
- 文件读写安全。
- 命令执行沙箱。
- prompt injection 防护。
- Memory / Skill 写入治理。
- 插件权限声明。
- 日志脱敏和审计。
- checkpoint / rollback。

这类系统不能只靠“模型听话”。必须把安全做成运行时能力：在工具调用前检查、在文件写入前检查、在后台维护前检查、在跨平台消息发送前检查。

<h2 id="q-018">面试问题：文件安全、命令安全和 Tool Loop Guardrails 如何协同？</h2>

三者解决不同层面的风险：

1. **文件安全**

   阻止写入敏感路径，例如 SSH key、shell profile、凭证文件、系统配置、Agent 自身密钥目录。可通过 denylist、safe root、realpath 和 symlink 检查实现。

2. **命令安全**

   对 Shell 命令按只读、写入、网络、安装、危险、破坏性分类，高风险命令需要确认或禁止。

3. **Tool Loop Guardrails**

   检测重复失败、无进展循环、同一只读工具反复调用、工具参数反复错误，必要时给模型 warning、block 或 halt。

协同方式：

- 文件安全和命令安全负责“单次动作是否允许”。
- Tool Loop Guardrails 负责“一串动作是否陷入坏模式”。
- 审批系统负责“高风险但用户明确需要的动作是否放行”。
- Trace 负责事后复盘和评测。

面试中可以举例：Agent 连续 5 次用相同参数调用 `search_files` 且无新信息，应触发 no-progress guardrail；Agent 试图写 `~/.ssh/authorized_keys`，应由 file safety 直接阻断。

<h2 id="q-019">面试问题：为什么 Worktree 隔离、Checkpoint 和 Rollback 对 Agent 很重要？</h2>

Agent 会自动编辑文件、运行命令、安装依赖、生成代码。没有隔离和回滚，错误动作会直接污染用户工作区。

**Worktree 隔离**：

- 每个 Agent 会话在独立 Git worktree 中工作。
- 多个 Agent 可并行探索。
- 避免覆盖用户未提交改动。
- 便于比较和选择最终 patch。

**Checkpoint**：

- 在高风险操作前保存文件系统快照或 Git 状态。
- 记录工具调用前后的 diff。
- 支持恢复到安全点。

**Rollback**：

- 当 Agent 改错、测试失败、用户不满意或命令产生副作用时恢复。
- 可按任务、文件或 checkpoint 回滚。

工程原则：

- 只读任务不需要重 checkpoint。
- 写文件、批量替换、依赖升级、格式化、迁移脚本前应 checkpoint。
- 回滚操作本身也要审计。
- 对生产 API 副作用不能只靠本地 rollback，需要补偿事务。

<h2 id="q-020">面试问题：多平台 Agent 的可观测性和本地行为画像如何设计？</h2>

多平台 Agent 的可观测性要同时服务 debug、安全审计、成本治理和用户体验优化。

建议记录：

- session / user / platform / channel / workspace。
- 模型和 provider。
- slash command。
- 工具调用和结果。
- Skill 加载和使用。
- Memory 读写。
- Context compression。
- 任务状态和后台事件。
- 权限审批。
- 成本、token、延迟。
- 错误和重试。
- 插件触发事件。

本地行为画像可以用于：

- 展示用户 Agent 使用情况。
- 发现高价值技能。
- 识别低效工具链。
- 生成成就或使用洞察。
- 辅助 Curator 判断 Skill 生命周期。
- 构造个人化推荐。

注意：

- 行为画像必须本地优先或租户隔离。
- 敏感内容脱敏。
- 用户可关闭、导出、删除。
- 不应把娱乐化指标凌驾于安全和隐私之上。

---

<h1 id="q-021">6. Agent 轨迹数据如何用于训练、评测和自我改进？</h1>

Agent 轨迹数据是指模型在执行任务时产生的完整对话、推理、工具调用、工具结果、环境状态和最终输出。它既可以用于评测，也可以用于训练下一代工具调用模型。

典型用途：

- 失败分析。
- Eval Harness 样本构造。
- SFT 数据生成。
- RL rollout。
- Tool Call Parser 训练。
- Reward Function 设计。
- Skill 发现。
- 工具 schema 优化。
- 用户行为洞察。

轨迹数据应包含：

- messages。
- tool calls。
- tool results。
- reasoning 或 scratchpad。
- environment state。
- final answer。
- reward / score。
- failure tags。
- model / prompt / tool version。

注意：轨迹数据高度敏感，包含用户输入、工具结果和潜在密钥，必须脱敏、授权和权限隔离后才能用于训练。

<h2 id="q-022">面试问题：Atropos / RL 环境对工具调用 Agent 有什么启发？</h2>

Hermes Agent 的 Atropos 环境体现了一种重要趋势：把工具调用 Agent 放进可交互环境中做 rollout、评分和训练，而不是只训练单轮问答。

通用架构：

```text
Dataset Item -> Prompt -> Agent Loop -> Tool Calls -> Environment State -> Reward -> Training / Eval
```

关键组件：

- BaseEnv：管理 worker、server、日志和评测。
- Agent Loop：多轮调用模型和工具。
- ToolContext：让 reward function 访问同一个 sandbox 状态。
- Terminal / Browser / File 工具：提供真实交互能力。
- Reward Function：运行测试、检查文件、比较状态。
- Tool Call Parser：把模型原始输出转成结构化工具调用。

启发：

- Agent 训练必须关注工具执行后的状态。
- Reward 不应只看文本答案。
- 同一个 task_id 要绑定同一个 sandbox。
- 评测环境和训练环境最好共用底层工具抽象。
- 多模型 tool calling 格式不统一，需要解析层。

<h2 id="q-023">面试问题：为什么不同模型需要不同 Tool Call Parser？</h2>

不同模型和推理服务的工具调用格式并不完全一致。

常见差异：

- OpenAI 风格 structured `tool_calls`。
- XML / ChatML `<tool_call>`。
- JSON 数组。
- Mistral `[TOOL_CALLS]`。
- Qwen / DeepSeek / Kimi / GLM 等各自格式。
- 原始 `/generate` endpoint 只返回文本，需要客户端解析。

Tool Call Parser 的作用：

- 从原始文本中提取工具名和参数。
- 处理 malformed JSON。
- 保留普通文本和工具调用边界。
- 为 Agent Loop 提供统一结构。
- 为训练生成 token mask 和 reward 对齐。

工程风险：

- 解析错误会导致工具误调用。
- 宽松解析可能被 prompt injection 利用。
- 多工具调用顺序必须保留。
- 解析失败要有 fallback 和失败标签。
- Parser 需要随模型版本更新。

面试中可以说：Tool Call Parser 是模型输出协议和 Agent Runtime 之间的适配层，决定了原始模型是否能可靠进入工具调用闭环。

<h2 id="q-024">面试问题：如何回答“请设计一个自进化、多平台、可长期运行的 Agent 系统”？</h2>

可以按九层架构回答：

1. **入口层**

   CLI、TUI、Web、IM、Email、Webhook、Cron、API Server。

2. **Gateway 层**

   平台适配、消息归一化、session 路由、命令注册、事件分发、权限回调。

3. **Runtime 层**

   模型循环、工具调用、interrupt、resume、streaming、task state、background task。

4. **Provider Router 层**

   多模型接入、能力路由、fallback、凭证解析、成本统计。

5. **Tool / Skill / Plugin 层**

   Tool Registry、Skill Hub、Plugin System、MCP Client、企业工具网关。

6. **Memory / Context 层**

   长期记忆、session search、context engine、compression、user profile、project context。

7. **Self-improvement 层**

   后台复盘、Skill 生成、Curator、知识评分、合并、归档、回滚。

8. **Safety / Governance 层**

   文件安全、命令安全、tool loop guardrails、sandbox、checkpoint、approval、audit。

9. **Eval / Training 层**

   trace、trajectory、harness、reward function、RL environment、失败回流。

关键原则：

- 入口和运行时解耦。
- 工具和技能注册化。
- 自进化只作用于受控知识资产。
- 后台复盘使用受限工具集。
- 所有副作用可审计、可回滚。
- 生产写操作默认需要审批。
- 轨迹数据既用于评测，也用于训练，但必须脱敏和授权。
- 通过 Harness 验证自进化是否真的提升，而不是只相信自我评价。

高分总结：

自进化多平台 Agent 的核心不是“接很多平台、装很多工具”，而是构建一个长期运行的智能操作系统：它能接收多端任务、稳定执行、沉淀经验、维护知识资产，并在严格安全和评测闭环中持续改进。

---

## 高频速记

1. 自进化 Agent 不只是记忆用户，而是把经验沉淀为 Memory、Skill、Policy 和 Eval Case。
2. 自进化必须可审计、可回滚、可约束，不能让 Agent 随意改自己。
3. 多平台 Gateway 把 CLI、TUI、IM、Email、Webhook、Cron 等入口统一到同一 Agent Runtime。
4. 多平台 session 必须按 platform、workspace、channel、user、agent、task 隔离。
5. 长期运行 Agent 需要 interrupt、resume、cron、background task 和 task state。
6. Provider Router 解决多模型接入、能力路由、fallback 和成本治理。
7. Tool Registry 解决工具 schema、handler、toolset、check_fn、风险等级和动态刷新。
8. Skill Hub 不是 prompt 集市，而是带依赖、权限、版本和生命周期治理的能力分发系统。
9. Plugin System 可扩展平台、工具、记忆、上下文引擎、可观测性和 Dashboard。
10. Curator 是后台知识维护器，负责评分、合并、归档和报告。
11. 后台复盘 fork 应限制工具集，只做知识整理，不做无人监督执行。
12. 文件安全、命令安全和 Tool Loop Guardrails 分别控制单次动作和行为模式。
13. Worktree、Checkpoint、Rollback 能降低自动编辑和命令执行对用户工作区的破坏风险。
14. Agent 轨迹数据可用于评测、失败回流、SFT、RL 和工具调用模型训练。
15. 不同模型工具调用格式不同，Tool Call Parser 是模型输出协议和 Agent Runtime 的适配层。

## 参考资料

- Nous Research, [**Hermes Agent README**](https://github.com/NousResearch/hermes-agent).
- Hermes Agent, [**v0.12.0 Curator Release Notes**](https://github.com/NousResearch/hermes-agent/blob/main/RELEASE_v0.12.0.md), 2026.
- Hermes Agent, [**Development Guide / AGENTS.md**](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md).
- Hermes Agent, [**Atropos Environments README**](https://github.com/NousResearch/hermes-agent/blob/main/environments/README.md).
- agentskills.io, [**Agent Skills Open Standard**](https://agentskills.io/).
