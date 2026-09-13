---
name: human-tone-polish
description: Polish Chinese technical, educational, and public-facing drafts to reduce mechanical or template-like AI tone while preserving the author's voice, technical accuracy, inline citations, figures, captions, and article structure. Use when the user asks to 去 AI 味、人味儿润色、打磨文章、精简啰嗦段落、改掉模板话术、优化技术文章语气、保留作者风格、润色正稿或润色稿，including but not limited to WeChat public-account, blog, course note, research explainer, and MagicAI/魔方AI空间 article workflows.
---

# Human Tone Polish

## Purpose

Polish Chinese technical and explanatory drafts so they read like the author's own article, not a generic AI rewrite.

Prioritize, in order:

1. Preserve technical correctness.
2. Preserve the author's position and phrasing habits.
3. Remove mechanical rhythm, template transitions, and over-explained prose.
4. Improve public-reading rhythm.
5. Keep citations, image placeholders, figure numbers, and tables coherent.

## Workflow

1. **Read the draft and nearby context.**
   - Inspect the target article first.
   - If available, inspect the outline, collaboration draft, final draft, user-edited version, image prompt file, or reference article in the same topic folder.
   - Treat the latest user-edited draft as the strongest style signal.

2. **Decide how much to touch.**
   - For a requested section polish, edit only that section unless the surrounding transition clearly breaks.
   - For a whole-article polish, keep the structure unless the draft has duplicated or obviously slow sections.
   - Prefer small, visible improvements over rewriting every paragraph.

3. **Apply the human-polish pass.**
   - Use [references/rewrite-principles.md](references/rewrite-principles.md) for editing decisions.
   - Keep direct technical sentences when they already work.
   - Remove writing-process phrases and empty scaffolding.
   - Shorten repeated examples, caveats, and conclusion padding.

4. **Apply the final check.**
   - Use [references/ai-tone-checklist.md](references/ai-tone-checklist.md) before delivery.
   - Check headings, image numbering, captions, links, and Markdown rendering.
   - Run a lightweight diff or Markdown check when files are edited.

5. **Learn from useful feedback.**
   - When the user flags a phrase, rhythm, heading, or rewrite as AI-like, decide whether the feedback is reusable beyond the current article.
   - If it is reusable, update the relevant reference file in this skill during the same turn.
   - Store style rules, not one-off facts. Do not add article-specific names, claims, links, or temporary preferences as permanent rules.
   - Keep each new rule concrete enough to guide a future edit.

## Editing Rules

- Do not make the prose uniformly polished. Some asymmetry is author voice.
- Do not turn every section into the same pattern.
- Do not add invented scenes, emotions, or personal anecdotes.
- Do not remove a caveat if the remaining technical claim becomes misleading.
- Do not rewrite a clear definition just to make it sound more stylish.
- Do not add slogans such as `未来可期`, `全面赋能`, `革命性突破`, or `生态闭环` unless the author explicitly wants that tone.

## Output

When editing files:

- Create `-润色稿.md` when polishing an existing `-正稿.md`, unless the user asks to overwrite.
- Preserve existing image references and placeholders.
- Briefly summarize the main changes and any claims that still need source verification.

When giving inline suggestions:

- Provide the improved sentence or paragraph directly.
- Explain only the key reason, such as `更短`, `少一点模板感`, `技术边界更稳`.
