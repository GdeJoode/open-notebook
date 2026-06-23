# Extraction pipeline — model & prompt decisions (naslagwerk / verantwoording)

**Date**: 2026-06-23 · **Scope**: how the entity-extraction recipe (model + prompt +
chunking) was chosen for the Dutch Regio-Deal / government corpus. Written as a
reference + justification for the choices, after an extensive empirical search.

---

## 1. The problem we were solving

Two coupled quality problems on the live KG (2,357 entities, Regio-Deal corpus):

1. **Typing**: 27% of entities were `other` and 44% were generic `concept`/`topic`.
   Root cause (Track L): the LLM extracted rich Dutch domain types (`Gemeente`,
   `Ministerie`, `RegioDeal`, `BeleidsThema`) but persistence flattened them onto a
   20-type generic enum with no ontology bridge. **Fixed by Track L** (ontology→canonical
   bridge + `primary_type`/`type_tags` preservation + content-driven `policy_themes`
   selection + EN/NL aliases).
2. **Recall**: extraction under-produced — a document yielding ~100-300 mentions
   surfaced only a handful of entities once we tried to reduce LLM call-count via
   context-packing. This document is mainly about **2** (the model + prompt + chunking
   recipe that fixes recall while keeping the rich typing).

**Filtering philosophy (user decision, load-bearing)**: extract **exhaustively** (recall
over precision), then reduce the count **deterministically** (dedup, merge) + **manual
review** (the K.6 resolution UI). Do NOT let the LLM black-box-filter / summarise — the
operator filters deliberately, not the model silently.

---

## 2. The model journey (what we tried and why)

Measured on one Regio-Deal doc (Convenant Midden-Limburg), L.1-L.4 typing pipeline +
Track M context-packing held constant where noted.

| Model | Platform | Recall (entities) | Typing | Speed | Verdict |
|---|---|---|---|---|---|
| **mistral-medium** | NIM | — | (summaries) | slow | Too slow for high-volume extraction (kept for summarisation) |
| **llama3.1:8b** | local Ollama | generic | **0% rich** (Organization/Person/Concept) | fast (~min) | **8B too small** — emits generic English labels, ignores the ontology |
| **qwen2.5:14b** | local Ollama | rich (200+ raw) | rich | **60+ min, STALLED** | Followed the ontology but impractically slow locally + malformed JSON |
| **Gemini 2.5 Flash** | Google free | rich | rich | n/a | **Free tier = 20 requests/DAY** (`PerDay`); one doc needs ~3-28 calls → unusable free. Per-minute trip too. Needs billing. |
| **Mistral Large 3 (675B)** | NIM | rich (Pass-1) | rich | slow (overkill) | **NIM-Mistral structured-output blocked**: `guidance` backend unsupported for the Mistral tokenizer → Pass-2 aborts. 675B is also overkill/slow. |
| **Llama 3.3 70B** | NIM | **8 (under-extracts)** | rich (BeleidsPijler/Indicator, 0% other) | slow (~7-9 min) | Types well but **under-follows the "extract all" instruction**; later flaky (RuntimeError on NIM). |
| **ministral-3:14b** ✅ | **Ollama Cloud** | **383 → 311 persisted** | **rich, 0% other, 100% primary_type** | ~9 min (3 windows) | **THE WINNER** — clean 14B instruct, follows the exhaustive prompt, rich domain typing, 264 relations. |

### The winning run (ministral-3:14b, Ollama Cloud, exhaustive prompt, cap=6000)
- **383 entities extracted → 313 after deterministic merge → 311 persisted** (237 active
  for the doc). 0% `other`, 100% `primary_type`. **264 relations** (the KG previously had 3).
- Types: `BeleidsThema`(60), `BeleidsPijler`(51), `Indicator`(31), `Programma`(14),
  `Person`(11), `Budget/Amount`(10), `Organisatie`(10), `Overheidsorgaan`(9),
  `Gemeente`(7), `Ministerie`(5) — the full domain ontology.

---

## 3. Key findings

1. **The typing pipeline (Track L) works — IF the model uses the ontology.** The bridge
   maps `Gemeente`→administrative_area, `BeleidsThema`→topic correctly. But it can only
   map what the model emits: a weak model (llama-8b) emits generic English labels →
   generic types. **Model instruction-following is the lever for rich typing.**

2. **Recall scales with LLM-call count / per-window exhaustiveness, NOT window size.**
   Packing the whole document into 1 big call → the model **summarises** (8 entities).
   The model extracts a roughly bounded set per call, so you need several moderate windows
   for completeness. This **overturned the naive Track M premise** ("pack big = fewer
   calls, same recall"). Sweet spot found: **cap≈6000 tokens/window** (3 windows for this
   doc) — small enough that the model stays exhaustive, few enough to beat 14-28 mini-calls.

3. **The extraction prompt was too weak.** "Extract all instances" was ignored. The
   **exhaustive prompt** (N.1) — "extract EVERY entity, sentence by sentence, dozens
   expected, do NOT summarise, do NOT filter (downstream + manual does that), map
   aggressively to the schema, use `other` only as a never-drop fallback" — combined with
   a model that follows instructions, took recall from ~8 to 311.

4. **Model size matters but 14B is enough.** llama-8b = too small (generic). qwen-14b /
   ministral-14b = the size where the ontology is followed. The huge models (675B, 397b,
   120b) are overkill + slow for documents this size. **Mid-size 14B instruct is the
   sweet spot.**

5. **Cloud free-tier shape matters more than headline specs.** Gemini's "1500/day"
   marketing was actually **20/day (PerDay)** on this project — one document exceeds it.
   Ollama Cloud's tier **resets** (5h/weekly) and exposes the same OpenAI-compatible API
   as local Ollama → cheapest integration + sustainable.

---

## 4. The chosen recipe (production)

- **Model**: `ministral-3:14b` via **Ollama Cloud** (provider `ollama-cloud`,
  `https://ollama.com/v1`, `OLLAMA_CLOUD_API_KEY`). Mid-size, instruct, Dutch-capable,
  reset-tier. **Local llama as the failover fallback** (Track J chain).
- **Prompt**: exhaustive extraction + aggressive schema mapping + `other` fallback
  (N.1 — `prompts/pass2.py` + `ontology_manager/prompts.py`).
- **Chunking**: `EXTRACTION_MAX_WINDOW_TOKENS=6000` (Track M.3 cap) — moderate windows
  that preserve recall.
- **Typing**: Track L (ontology→canonical bridge, content-driven `policy_themes`,
  EN/NL aliases, canonical-passthrough).
- **Reduction**: deterministic merge + the K resolution layer (dedup/aliases/overlay)
  + the K.6 review UI — operator-controlled, not LLM-filtered.

---

## 5. Infrastructure findings (gotchas worth recording)

- **NIM + Mistral structured output**: NIM's default `guidance` guided-decoding backend
  is unsupported for the Mistral tokenizer (`use xgrammar/outlines or tokenizer_mode=hf`)
  → Pass-2 aborts. NIM Llama tokenizers are fine. To use a NIM Mistral with structured
  output you'd thread `nvext.guided_decoding_backend` — not exposed via esperanto's path.
  (Ollama Cloud uses its own format and avoids this.)
- **Rate-limit classification**: esperanto wraps a provider 429 / `RESOURCE_EXHAUSTED`
  as a plain `RuntimeError`; `default_is_rate_limit` was extended (Track M interim) to
  match the message so a cloud 429 paces + backoff-retries on the same provider instead
  of an immediate failover-to-local.
- **Per-provider RPM** (Track M.2): `google≈6`, `nvidia≈30`, `ollama` unlimited,
  env-overridable (`GOOGLE_RPM`/`NVIDIA_RPM`/`OLLAMA_RPM`).
- **Worker hang on cancel**: cancelling a job's DB row does NOT kill the in-flight worker
  thread (stuck LLM call) — a container restart is the reliable way to clear it.
- **Job status lag**: `job.status` stays `queued` during a run; detect completion via the
  `Entity extraction completed for source ...` log line or the persisted-entity count.

---

## 6. Relations — RESOLVED (Track O.1 + migration 58)

The KG had a long-standing **3-relation floor** despite extraction producing hundreds.
Two root causes, both fixed:
1. **Endpoint typing mismatch**: the relation step typed endpoints via the alias-only
   `_normalize_entity_type`, while entities are typed via the L.1 bridge — so the
   `(name, type)` RELATE lookup missed for bridge-only types (`Indicator`→topic vs
   other). Fixed: relation endpoints now bridge-resolve (carried `source_type`/`target_type`
   wins, else the bridge-resolved `type_by_name` map), with a **name-only fallback** on a
   typed miss (K.7a cross-type homograph safety preserved). Plus a RELATE record-link bug
   (`SELECT VALUE id` → string; coerced back with `type::thing`).
2. **Live schema drift**: the live `relation` table was a pre-migration-39 **NORMAL**
   table, so migration 39's `TYPE RELATION` never took effect and every `RELATE` failed
   ("not a relation, expected a NORMAL"). **Migration 58** drops the malformed legacy table
   (3 null-endpoint rows) and re-asserts the edge `TYPE RELATION` table.

**Verified**: 3 → **261 relations** on one doc, with meaningful Dutch domain predicates
(`IS_PIJLER_VAN`, `LEIDT_TOT`, `VERSTERKT`, `VERMINDERT`, `BIJDRAGT_AAN`). The KG now has a
real graph.

## 7. Open / deferred items
- **Track M.4(a)**: full per-document failover re-chunking (the interim guard bounds
  `num_ctx` + logs but does not re-split a head-sized prompt for a smaller-context
  fallback — Ollama truncates it).
- **NIM-Mistral structured output**: a `guided_decoding_backend` override would unlock NIM
  Mistral models if ever needed.
- **Cap tuning**: 6000 was the chosen sweet spot; 4000/8000 unmeasured — revisit if recall
  or speed needs shifting.
- **Track N formalisation**: the exhaustive prompt (N.1) + Ollama Cloud wiring (N.2) are on
  `track/n-exhaustive-prompt`, adversarially reviewed before merge.
