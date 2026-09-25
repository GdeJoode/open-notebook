# D6 judge prompt (verbatim)

Five judge instances, model `sonnet` (Claude Sonnet 5), run 2026-09-25 as Claude Code
subagents, one per batch. Each saw only the files named in its prompt: the full
document text and a pairs file with `item`, `entity_a`, `type_a`, `entity_b`,
`type_b`. No origin, relation type or score.

| judge | documents (judge_batches/) | pairs |
|---|---|---|
| A | hjrb1cjvzv86oyaojblj | 52 |
| B | 9q3avwz6er6413tqrg6t | 15 |
| C | 052dtl7jrwu1czlpnui4 | 13 |
| D | 0o96pew25fovpoom5n6r, dndibxmjveoxk7tfqfsl | 11 + 8 |
| E | 1k3c41i456l7ywu6y7iv, niqnh6775y0xxwpz0nzd | 3 + 2 |

The prompt, with `<FILES>` and `<OUTPUT>` being the only parts that differed per
judge (D and E additionally received: "two separate documents, each with its own
pairs; judge each pair ONLY against its own document"):

```
You are a careful annotator judging whether a document supports a relationship between two named entities. You will work on these files in E:\repos\private\open-notebook\claudedocs\d6-edge-prediction\judge_batches\ :

<FILES>

Read the WHOLE document first (use the Read tool, in several parts if needed). Then, for EVERY line in the pairs file (each has "item", "entity_a", "type_a", "entity_b", "type_b"), decide ONE verdict about what THIS DOCUMENT says about A and B:

- "direct": the document states, or clearly and specifically implies, a relationship between A and B THEMSELVES. Examples: A is part of / located in B; A leads, signs on behalf of, funds, is responsible for, or is a party to B; A and B are named as specifically working together on something; A is an instance or component of B.
- "indirect": A and B are connected ONLY through something shared — both appear in the same list of parties/signatories/members, both belong to the same region or deal, both relate to a common third entity — with no statement about A and B relating to each other.
- "none": the document contains no relationship between them (including: one of them does not occur at all, or they only occur in unrelated places).
- "unclear": you genuinely cannot decide.

Names in the text may differ in form from the entity names (e.g. "Alkmaar" for "Gemeente Alkmaar", abbreviations, inflections). Match sensibly, but do not invent. Judge only from the document, not from your world knowledge.

For "direct" and "indirect" you MUST give a verbatim quote (max ~300 characters) from the document that shows it. For "none"/"unclear" give a one-sentence reason.

Write your output with the Write tool to <OUTPUT> — one JSON object per line, one per pair, in this exact shape:
{"item": "E012", "verdict": "direct|indirect|none|unclear", "quote": "...", "reason": "one short sentence"}

Every item in the pairs file must get exactly one line. Do not modify any other file. Reply with only the count of each verdict.
```
