# AI Tone Checklist

Use this checklist after editing only the changed parts, then scan the whole article once.

## Mechanical Phrase Search

Search for these phrases and fix only when they sound templated:

- `接下来`;
- `下面`;
- `值得注意的是`;
- `需要指出的是`;
- `总而言之`;
- `未来可期`;
- `赋能`;
- `打造`;
- `颠覆`;
- `革命性`;
- `生态闭环`;
- `压住`;
- `撞到`;
- `先别急着`;
- `换成更口语`;
- `分工`;
- `触及`;
- `继续往前走`;
- `继续往下看`;
- `换个角度`;
- `视角收回来`;
- `相信很多`;
- `它们并不是一组并列术语`;
- repeated `有的负责`;
- `可以压成`;
- `一句话收束`;
- `核心思想很直接`;
- `思路很清晰`;
- `逻辑很简单`;
- repeated `放到`;
- `放到模型里`;
- `放进环境里`;
- `换 backbone`;
- `可适配的开发入口`;
- `行为塑形`;
- `训练塑形`;
- `真正的价值不在于`;
- repeated `不是...而是...`.

## Section Checks

- Does the opening reach the topic quickly?
- Does each heading sound like a public article, not an outline note?
- Are adjacent headings too similar?
- Is an early table or figure doing real work?
- Are figure placeholders numbered continuously?
- Do captions explain what the image shows?
- Are source links placed near the claims they support?

## Sentence Checks

For every changed sentence, ask:

- Did the meaning stay the same?
- Did a technical boundary disappear?
- Did the author's voice become too smooth or promotional?
- Did the sentence become longer just to sound polished?
- Can a concrete verb replace a vague word?

## Common Repairs

| Problem | Repair |
| --- | --- |
| `不是 X，而是 Y` appears repeatedly | State the positive claim directly |
| A section explains its own writing plan | Delete the plan sentence and start with the technical point |
| A paragraph gives two similar examples | Keep the sharper example |
| A caveat becomes a long mini-section | Keep one boundary sentence |
| A title asks a vague question | Use a layer + topic title when it fits, such as `数据：真实、合成、人类视频为什么要一起用？` |
| A table intro sounds like notes | Replace with a reason the table helps the reader |

## Delivery Checks

- No duplicated paragraphs.
- No broken Markdown tables.
- No accidental loss of image paths or placeholders.
- No made-up citations or facts.
- File name matches the requested stage.
