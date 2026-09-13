# Rewrite Principles

Use these principles when polishing Chinese technical, educational, research-explainer, blog, and public-account articles.

## 1. Keep The Person In The Sentence

Before editing a sentence, ask what it is doing:

- making a technical claim;
- showing the author's judgment;
- connecting to a previous article;
- naming a boundary;
- helping the reader scan.

If the sentence already does its job, do not rewrite it only because it could be smoother.

Preserve:

- the author's chosen examples;
- short direct judgments;
- mild口语 expressions that sound natural;
- uneven but readable rhythm;
- concrete technical nouns.

Avoid adding:

- decorative metaphors;
- invented scenes;
- grand claims;
- overly neat three-part structures;
- polished but empty transitions.

## 2. Cut Writing-Process Language

Published prose should not expose the drafting process.

Replace or delete phrases like:

- `下面我们来看`;
- `接下来进入`;
- `可以先用一句话概括`;
- `把主线压住`;
- `用一句话收束`;
- `这篇文章不是...而是...`;
- `先别急着...`;
- `换成更口语的版本`;
- `这里有一个关键判断`;
- `真正要看的是什么`.
- `这次把视角收回来`;
- `换个角度`;
- `继续往下看`.

Prefer direct article language:

- `核心问题是...`;
- `这一步解决...`;
- `它对应的是...`;
- `更准确地说...`;
- `这也是...的分界线`;
- `工程上最容易卡住的是...`.

For article openings and section transitions, avoid describing the writing action itself. Sentences such as `换个角度看` or `把视角收回来` often sound like outline commentary. Prefer starting from the object directly: `这一篇从 X 讲起`, `X 背后有一条开发链路`, or `把 X 拆开，会看到...`.

In openings, avoid broad filler setups such as `相信很多人...`, `你可能经常看到...`, or `它们并不是一组并列术语` when the article can enter the topic directly. Prefer a concrete entry sentence such as `学习 X，绕不开这些概念：...`, then explain the relationship in one compact paragraph.

Avoid vague evaluative padding before a technical claim, such as `核心思想很直接`, `思路很清晰`, or `逻辑很简单`. If the next sentence already states the technical point, remove the padding and write `核心思路是：` or start with the claim directly.

When moving from intuition to architecture, avoid loose bridge phrases such as `放到模型里`. Prefer precise but readable connectors such as `对应到架构上`, `具体到模型结构`, or directly name the modules.

When polishing technical prose, do not keep casual English-Chinese shortcuts if a precise Chinese term is available. For example, replace `换 backbone` with `更换模型主干` unless the sentence is quoting code, a model name, or an established paper term. Also avoid abstract product-like phrases such as `可适配的开发入口`; make the adaptability concrete with terms like `可微调、可接入、可部署`.

When explaining evaluation or deployment, avoid loose phrases such as `放进环境里` if the point is interaction or execution. Prefer concrete wording such as `进入环境交互`, `在环境里连续执行`, or `连续跑起来`, depending on the article rhythm.

Watch for overused all-purpose verbs such as `放到`. It is often a sign that the sentence has not named the real technical relationship. Replace it with the specific action when possible: `接入` for integration, `统一到` for interface or format convergence, `映射到` for representation conversion, `封装成` for packaging, `对齐到` for schema or protocol matching, and `纳入` for scope inclusion.

Avoid stiff translated abstractions such as `行为塑形` or `训练塑形` in public-facing Chinese drafts. Unless the article is explicitly discussing behavioral psychology or RL shaping as a technical term, write the concrete training effect instead: `训练模型按要求输出`, `调整模型的输出习惯`, `让模型更会遵循任务和偏好`, or `训练与对齐`.

## 3. Make Technical Explanation Shorter, Not Thinner

For technical paragraphs, remove repetition before removing substance.

Keep:

- input / output;
- model modules;
- training or inference signal;
- engineering constraint;
- one necessary boundary.

Cut:

- the second example if the first has already proved the point;
- usage lists after a definition is already clear;
- long caveat paragraphs;
- repeated `价值在于`;
- broad claims without a mechanism.

Useful compact shape:

```text
概念 / 工作是什么
它接收什么
它输出什么
关键机制是什么
边界在哪里
```

Do not force this shape on every section. Use it as a compression guide.

## 4. Preserve Public Reading Rhythm

Prefer short paragraphs, but do not make every paragraph one sentence.

Good public-facing rhythm usually alternates:

- one direct claim;
- one explanation sentence;
- one example or table;
- one boundary sentence.

Avoid:

- long paragraph blocks on mobile;
- consecutive sections that start with the same grammar;
- repeated `不是 X，而是 Y`;
- repeated `A 负责...B 负责...C 负责...`;
- long `有的负责...有的负责...` enumerations in openings;
- stacked bold sentence starters.

## 5. Handle Examples Carefully

Examples should clarify, not decorate.

Keep one concrete example when it explains the concept quickly.

Delete examples when they:

- repeat the same boundary;
- pull the reader into another domain too early;
- sound like a generic assistant-generated scenario;
- make a short glossary section feel like a mini essay.

## 6. Keep Source-Linked Claims Grounded

When a sentence cites a paper, repo, product page, or official blog:

- keep the link near the relevant claim;
- avoid making a first-party product page sound like independent benchmark evidence;
- separate model release facts from performance claims;
- avoid saying `证明` unless the source truly proves the exact claim.

For fast-changing AI systems, prefer wording such as:

- `官方资料显示`;
- `GitHub 当前给出的信息是`;
- `更像一个开发入口`;
- `可以支持这个判断`;
- `不能直接等同于所有真实部署场景`.
