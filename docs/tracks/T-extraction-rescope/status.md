# Track T — status

## T.1 — extraction-economy baseline — DONE; Track PARKED (2026-06-27)

**Verdict (decision gate): stop thin.** The measurement (`extraction-economy.md`) showed:
- R.6 already captures the search win — it drops 94% of active entities from the search signal;
  <0.6% of the LLM's output reaches ranking.
- The remaining ~67% "wasted-for-search" output is NOT free to cut: those generic/topic rows
  feed the Obsidian/NetworkX/JSONL exports, and the locked decision is to keep exports rich.
- So, under the chosen course (rich exports + careful, search-projection-only), there is no safe
  net win left. The only remaining win is extraction-cost (T.2b prompt change), which conflicts
  with rich exports.

**Decision (user, 2026-06-27): PARK Track T.** Do not touch the load-bearing extraction pipeline.
LLM extraction stays where it is; R.6 owns the search-side cleanup; exports stay rich.

**Re-open condition:** if LLM extraction cost becomes a real pain point (larger corpus, NIM bill),
revisit **T.2b** as a deliberate "leaner exports for ~67% less extraction output" trade-off. The
T.1 numbers + the prompt origins (`prompts.py` `## Thematic Concepts` + the `pass2.py` exhaustive-recall
mandate) are the ready starting point.

T.2–T.5 remain drafted in `plan.md` but unstarted.
