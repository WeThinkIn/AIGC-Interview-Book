window.frontierHighlightsData = {
  header: {
    eyebrow: "Frontier Highlights",
    title: "把知识星球的高价值前沿精选，做成主页里的动态高光区",
    description:
      "这里将知识星球的季度精选浓缩为四个主题方向，既承接主页的知识地图，也补充更贴近行业变化的动态观察、能力判断与实战入口。",
  },
  hero: {
    label: "2026 Q1 精选内容",
    titleHtml:
      '用 <span class="frontier-word frontier-word-agent">AI Agent</span>、<span class="frontier-word frontier-word-multimodal">多模态</span>、<span class="frontier-word frontier-word-research">具身智能</span> 与 <span class="frontier-word frontier-word-model">前沿模型应用</span>，帮助求职者看清“行业现在在追什么”',
    description:
      "精选报告聚合全球最新 AI 技术动态、行业研报与落地方法，强调的不只是资讯密度，更是对能力边界、产品趋势、工程实践与职业判断的梳理。",
    pills: [
      { label: "10 条精选", tone: "signal" },
      { label: "4 大主题", tone: "neutral" },
      { label: "AI Agent 演进", tone: "agent" },
      { label: "多模态实战", tone: "multimodal" },
      { label: "硬核学术", tone: "research" },
      { label: "大模型应用", tone: "model" },
    ],
    metrics: [
      {
        label: "核心目标",
        valueHtml:
          '帮助用户快速捕捉 <span class="frontier-word frontier-word-agent">技术拐点</span> 与 <span class="frontier-word frontier-word-model">岗位新要求</span>',
      },
      {
        label: "内容特征",
        valueHtml:
          '资讯筛选 + 原理拆解 + <span class="frontier-word frontier-word-multimodal">落地链接</span> + 职业判断',
      },
      {
        label: "适合人群",
        valueHtml:
          '求职者、算法工程师、<span class="frontier-word frontier-word-multimodal">应用开发者</span> 与持续跟踪 AI 方向的人',
      },
    ],
  },
  themes: [
    {
      key: "agent",
      badge: "Theme 01",
      shortLabel: "Agent",
      count: "3 篇",
      title: "AI Agent 进化论",
      descriptionHtml:
        '聚焦 <span class="frontier-word frontier-word-agent">AI Agent</span> 技术突破与产业趋势，帮助理解模型能力边界、交互范式变化与未来岗位机会。',
      spotlight: [
        { label: "能力边界", tone: "agent" },
        { label: "交互范式", tone: "agent" },
        { label: "岗位机会", tone: "signal" },
      ],
      items: [
        {
          index: "01",
          overline: "趋势判断",
          title: "LLM/Agent 模型跨过临界点的迷茫期",
          summaryHtml:
            '剖析行业割裂现象，厘清 <span class="frontier-word frontier-word-agent">Agent 提效</span> 与岗位变化之间的关系，帮助建立更稳的技术与职业判断。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/45811251281585418",
        },
        {
          index: "02",
          overline: "架构拆解",
          title: "一文彻底搞懂 OpenClaw：原理·架构·Skills·部署",
          summaryHtml:
            '从原理到本地化部署完整拆解 OpenClaw，理解自然语言如何逐步成为新的 <span class="frontier-word frontier-word-agent">系统交互层</span>。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/22811421281248111",
        },
        {
          index: "03",
          overline: "行业研报",
          title: "《OpenClaw：吹响 AI Agent 时代号角》行业研报",
          summaryHtml:
            '围绕 Agent 市场规模、算力需求、Token 消耗与国产大模型投资机会给出 <span class="frontier-word frontier-word-signal">宏观视角</span>。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/45811251585128458",
        },
      ],
    },
    {
      key: "multimodal",
      badge: "Theme 02",
      shortLabel: "Multimodal",
      count: "4 篇",
      title: "多模态实战指南",
      descriptionHtml:
        '聚焦实时交互、多模态 Prompt、设计工作流与企业级落地方法，适合希望把模型能力接入 <span class="frontier-word frontier-word-multimodal">真实产品</span> 的人。',
      spotlight: [
        { label: "实时交互", tone: "multimodal" },
        { label: "Prompt 设计", tone: "multimodal" },
        { label: "生产落地", tone: "signal" },
      ],
      items: [
        {
          index: "04",
          overline: "实时智能体",
          title: "Google Gemini 3.1 Flash Live · 实时多模态智能体",
          summaryHtml:
            '覆盖低延迟语音到语音能力、官方 API 使用方式、定价与代码示例，适合快速搭建 <span class="frontier-word frontier-word-multimodal">实时交互链路</span>。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/55188458514411214",
          resources: [
            {
              label: "模型卡",
              url: "https://deepmind.google/models/model-cards/gemini-3-1-flash-live/",
            },
            {
              label: "开发文档",
              url: "https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview",
            },
            {
              label: "产品博客",
              url: "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-live/",
            },
          ],
        },
        {
          index: "05",
          overline: "工作流设计",
          title: "2026 多模态提示工程 & 设计工作流（Google Stitch）",
          summaryHtml:
            '面向产品与设计场景，展示如何借助自然语言快速生成界面原型与交互流程，加速 <span class="frontier-word frontier-word-multimodal">UX / MX 设计</span>。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/55188458514452554",
        },
        {
          index: "06",
          overline: "Prompt 框架",
          title: "Multimodal AI in 2026 · 如何改变 Prompt 写法",
          summaryHtml:
            '基于 GPT-4o、Gemini 等模型总结多模态 Prompt 框架，覆盖图像、音频、视频写法差异。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/55188458514415514",
        },
        {
          index: "07",
          overline: "企业落地",
          title: "企业级多模态落地指南 · Digital Transformation",
          summaryHtml:
            '聚焦从实验室 Demo 走向生产环境的关键步骤，覆盖数据治理、模型选型与实施路径。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/14588148451158812",
        },
      ],
    },
    {
      key: "research",
      badge: "Theme 03",
      shortLabel: "Research",
      count: "2 篇",
      title: "硬核学术资源",
      descriptionHtml:
        '面向希望补齐理论深度的人群，精选课程与综述类内容，把前沿研究与工程趋势连接起来。',
      spotlight: [
        { label: "扩散模型", tone: "research" },
        { label: "VLA 世界模型", tone: "research" },
        { label: "理论深度", tone: "signal" },
      ],
      items: [
        {
          index: "08",
          overline: "体系课程",
          title: "德克萨斯大学奥斯汀分校 · 扩散模型全体系教学",
          summaryHtml:
            '覆盖扩散模型原理、DDIM、蒸馏采样加速，以及 LoRA、ControlNet 等可控生成技术。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/55188458518221184",
        },
        {
          index: "09",
          overline: "前沿综述",
          title: "面向通用具身智能的 VLA 代理世界模型综述",
          summaryHtml:
            '梳理世界模型与视觉-语言-动作代理结合的主要范式，聚焦 <span class="frontier-word frontier-word-research">物理常识</span> 与安全性问题。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/55188458515854414",
          resources: [
            {
              label: "论文地址",
              url: "https://www.techrxiv.org/users/1019104/articles/1379248-towards-generalist-embodied-ai-a-survey-on-world-models-for-vla-agents",
            },
          ],
        },
      ],
    },
    {
      key: "model",
      badge: "Theme 04",
      shortLabel: "Model Ops",
      count: "1 篇",
      title: "前沿模型应用",
      descriptionHtml:
        '关注新一代大模型在真实业务中的使用方式，让用户看到模型能力如何转换为 <span class="frontier-word frontier-word-model">企业效率</span> 与个人生产力。',
      spotlight: [
        { label: "业务提效", tone: "model" },
        { label: "应用场景", tone: "model" },
        { label: "价值创造", tone: "signal" },
      ],
      items: [
        {
          index: "10",
          overline: "实战应用",
          title: "《GPT-5.4 实战应用完全指南（2026 年）》",
          summaryHtml:
            '梳理 GPT-5.4 在企业财务、专家咨询、电商运营等场景中的实战路径，强调效率提升与价值创造。',
          url: "https://wx.zsxq.com/group/48884124114188/topic/14588148454885452",
          resources: [
            {
              label: "电子书预览",
              url: "https://wx.zsxq.com/mweb/views/weread/search.html?keyword=GPT-5.4实战应用完全指南",
            },
          ],
        },
      ],
    },
  ],
  benefits: {
    label: "加入后可获得",
    titleHtml:
      '除了知识地图之外，再补上一层真正跟着 <span class="frontier-word frontier-word-signal">行业变化</span> 走的内容密度',
    items: [
      {
        title: "一手资讯",
        text: "跟进最新模型发布、技术架构更新与行业报告，避免只学旧知识。",
      },
      {
        title: "深度解析",
        text: "不仅转发信息，更拆技术原理、落地难点与背后的商业逻辑。",
      },
      {
        title: "实战工具",
        text: "获取可复用的 Prompt 框架、部署资料、方法论文档与案例入口。",
      },
      {
        title: "社群价值",
        text: "与同行交流，形成对技术变革、求职窗口与方向选择的共同判断。",
      },
    ],
  },
  cta: {
    label: "Knowledge Planet",
    titleHtml:
      '进入知识星球，持续追踪 <span class="frontier-word frontier-word-model">AI 前沿变化</span>',
    description:
      "如果主页负责建立系统框架，那么知识星球更像是动态更新层，帮你把技术趋势、应用案例与职业判断补齐。",
    bullets: ["动态追踪", "方法沉淀", "案例链接", "长期更新"],
    buttonText: "立即加入",
    buttonUrl: "https://t.zsxq.com/YtJ09",
    footnote: "提示：加入知识星球可获得更多内容。",
  },
};
