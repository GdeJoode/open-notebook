"""Every environment variable a service is given is read by something (PC.6).

Three knobs shipped here that configured nothing — `EXTRACTION_MODEL`,
`EXTRACTION_NUM_CTX` and `DEFAULT_PRIVACY` — and the first two cost a fully
reverted branch: a model was repointed in `docker-compose.yml`, the container
restarted, and the resolver went on returning what it always had.

The service-local AST guard cannot see this class, because the defect is the
RELATIONSHIP between a compose entry and the code: `docker-compose.yml` is the
file an operator edits, so a name that appears there and nowhere else is a
promise the system does not keep. This test is the compose half.

It asks only whether the name is referenced ANYWHERE — not whether the reference
is live. `EXTRACTION_MODEL` would have passed this and failed the AST guard; the
two are complementary and both are needed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "docker-compose.yml"

#: Names set in compose that nothing in the tree reads, with the reason. Consumed
#: by an image's own entrypoint, a third-party library, or the container runtime.
_ALLOWED: Dict[str, str] = {
    # Read outside the tracked tree, each verified rather than assumed. The list
    # was 17 entries and 13 of them were inert — pre-silencing names the guard
    # never asked about, which is how an allow-list stops meaning anything.
    "ESPERANTO_LLM_TIMEOUT": "esperanto/utils/timeout.py:18",
    "HF_HOME": "read by huggingface_hub",
    "SURREAL_EXPERIMENTAL_GRAPHQL": "read by the SurrealDB binary",
    "SURREAL_ROCKSDB_BLOCK_CACHE_SIZE": "read by the SurrealDB binary",
}

#: Files the scan must NOT read. `docker-compose.yml` is the question, not the
#: answer. The two guard sources are excluded because they NARRATE the knobs they
#: were written to catch — `DEFAULT_PRIVACY`, `EXTRACTION_MODEL` and
#: `EXTRACTION_NUM_CTX` all appear in their comments and allow-lists, so
#: re-adding any of the three to compose passed silently. The guard's own
#: documentation of the defect made the defect invisible to it, which is the
#: self-reference bug caught for the sentinel, not generalised.
_SELF_EXCLUDED = frozenset({
    "docker-compose.yml",
    "tests/test_compose_env_is_consumed.py",
    "services/extraction/tests/test_no_dead_env_constants.py",
})

_ENV_LINE = re.compile(r"^\s+-\s+([A-Z][A-Z0-9_]*)=", re.M)


def _code_only(source: str) -> str:
    """Python source with COMMENTS removed. String literals are kept.

    A name that appears only in a comment is not consumed by anything — and the
    comments EXPLAINING a knob's removal were what hid the removal from this
    guard: `EXTRACTION_MODEL` and `EXTRACTION_NUM_CTX` are named in
    `services/extraction/api.py`'s note about deleting them, so re-adding either
    to compose passed silently.

    Strings are KEPT, because an env-var name lives inside one —
    `os.getenv("DEFAULT_PRIVACY", …)`. Stripping them removed the very thing the
    read patterns match, and every name came back unreferenced.

    Falls back to the raw text on a syntax error: an unparseable file is a
    different problem, and dropping it from the scan would manufacture false
    "dead" verdicts.
    """
    import io
    import tokenize

    try:
        kept = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type != tokenize.COMMENT
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    return " ".join(kept)


def _compose_env_names() -> Set[str]:
    return set(_ENV_LINE.findall(COMPOSE.read_text(encoding="utf-8")))


#: How a name is actually READ from the environment, per file type. Matching the
#: read CONTEXT rather than the bare token is what makes the answer mean
#: something: a generous substring match reported `EXTRACTION_MODEL` as consumed
#: because `tests/test_derived_state_has_readers.py` binds a module constant of
#: that name to a FILE PATH — a coincidental collision, and the same trap PC.1b's
#: invariant took four rounds to escape.
_READ_PATTERNS = (
    # Whitespace-tolerant: `_code_only` re-joins tokens with spaces, so the
    # source `os.getenv("X")` reaches these as `os . getenv ( "X" )`. Anchoring
    # on `getenv\(` matched nothing and reported every name dead — a false
    # positive, which for this guard is the worse failure.
    # Any env-shaped reader, not just `os.getenv`. The services wrap it —
    # `_bool_env("DOCLING_DO_OCR", …)`, `_int_env`, `_str_env` — and anchoring on
    # `getenv` alone reported 22 genuinely-read names as dead. A false positive
    # here is the worse failure: it trains a reader to add allow-list entries.
    # Any env-shaped READER, not just `os.getenv`. The services wrap it —
    # `_bool_env("DOCLING_DO_OCR", …)`, `_int_env`, `_str_env` — and anchoring on
    # `getenv` alone reported 22 genuinely-read names as dead. A false positive
    # here is the worse failure: it trains a reader to add allow-list entries.
    #
    # `setenv`/`delenv` are excluded: a test that merely SETS a variable is not a
    # consumer of it, and counting them is the coincidental-collision class this
    # pattern set exists to close. No compose name relies on it today; the
    # exclusion is so none quietly starts to.
    r"\b(?!setenv\b)(?!delenv\b)\w*env\w*\s*\(\s*[\"']{name}[\"']",
    # A name bound to a constant and read through it — the shape at
    # `grobid_reference_service.py`: `_GROBID_URL_ENV = "GROBID_URL"`, then
    # `os.getenv(_GROBID_URL_ENV)`. Without this the guard cleared `GROBID_URL`
    # only because an unrelated test happens to spell it literally, so deleting
    # that test would have reported a live production knob as dead — the
    # false-positive direction this file names as the worse failure.
    r"_ENV\s*[:=][^\n=]*=\s*[\"']{name}[\"']",
    r"environ\s*\.\s*get\s*\(\s*[\"']{name}[\"']",
    r"environ\s*\[\s*[\"']{name}[\"']",
    r"\$\{{name}[:}]",
    r"\${name}\b",
)


def _referenced_names(names: Set[str]) -> Set[str]:
    """Which of ``names`` is read from the environment somewhere in the tree.

    Only the read contexts above count. That is stricter than "the token appears"
    and still deliberately generous about WHERE — a name read anywhere, including
    in a test, passes. The strict half is the AST guard in
    `services/extraction/tests/test_no_dead_env_constants.py`, which asks whether
    the value read is then used; the two are meant to be read together.

    **What neither can see**: a name that is genuinely read but whose value never
    applies. `DEFAULT_PRIVACY` is exactly that — `model_routing.py` reads it as the
    fallback for `defaults.default_privacy`, which the YAML always sets — so it
    passes here, correctly, and its removal from compose rests on the reasoning in
    that file rather than on a guard.
    """
    listed = subprocess.run(
        ["git", "ls-files", "--", "*.py", "*.yaml", "*.yml", "*.ts", "*.tsx", "*.sh"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    patterns = {
        name: [
            re.compile(p.replace("{name}", re.escape(name)))
            for p in _READ_PATTERNS
        ]
        for name in names
    }
    seen: Set[str] = set()
    for rel in listed:
        if rel in _SELF_EXCLUDED:
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if rel.endswith(".py"):
            text = _code_only(text)
        for name, regexes in patterns.items():
            if name in seen:
                continue
            if any(r.search(text) for r in regexes):
                seen.add(name)
    return seen


def test_every_compose_env_name_is_referenced_somewhere() -> None:
    names = _compose_env_names()
    assert len(names) > 20, f"walker control: only {len(names)} env names parsed"

    referenced = _referenced_names(names | {"OLLAMA_URL"})
    assert "OLLAMA_URL" in referenced, "detector control: a known-live name is missing"

    dead = sorted(n for n in names if n not in referenced and n not in _ALLOWED)
    assert not dead, (
        f"set in docker-compose.yml and referenced nowhere in the tree: {dead}. "
        f"An operator edits compose; a name that lives only there is a control "
        f"that configures nothing. Delete it, or wire it and say where."
    )


def test_the_allow_list_has_no_inert_entries() -> None:
    """Every entry must be doing work, or the list stops meaning anything.

    The first version carried 17 entries of which 13 were inert: names either not
    set in compose at all, or read somewhere the scan already sees. An allow-list
    that pre-silences questions nobody asked is where a real dead knob goes to
    hide.
    """
    names = _compose_env_names()
    referenced = _referenced_names(names)
    inert = sorted(n for n in _ALLOWED if n in referenced or n not in names)
    assert not inert, (
        f"allow-list entries that silence nothing: {inert}. Remove them — an "
        f"entry earns its place only by being set in compose AND unreadable from "
        f"the tracked tree."
    )


def test_the_removed_knobs_would_be_caught_if_re_added() -> None:
    """The three this guard exists for, checked by name.

    Round 2 shipped a version that could not catch any of them: the scan read the
    guards' own sources, which NARRATE the knobs, so `EXTRACTION_MODEL` and
    `EXTRACTION_NUM_CTX` were "referenced" by the comments explaining their
    deletion. The guard's own documentation of the defect made the defect
    invisible to it.

    `DEFAULT_PRIVACY` is deliberately NOT in this list: it is genuinely read
    (`model_routing.py` uses it as the fallback for `defaults.default_privacy`),
    and what made it dead in compose is that the YAML always sets that key — a
    subtler thing this guard cannot see and does not claim to. Its removal rests
    on the reasoning in that file.
    """
    caught = {"EXTRACTION_MODEL", "EXTRACTION_NUM_CTX"}
    assert _referenced_names(caught) == set(), (
        "a removed knob still reads as consumed — check whether a comment or an "
        "unrelated same-named constant is standing in for a real reader"
    )


def test_the_detector_would_catch_a_planted_name() -> None:
    """Mutant control: the scan must be able to fail.

    Without it, the test above passes equally well against a `_referenced_names`
    that returns every possible token — which is what a generous substring match
    is one bug away from.

    The sentinel is ASSEMBLED rather than written out: this file is itself
    scanned, so a literal would find itself and the control would pass for the
    wrong reason. It did, on its first run.
    """
    sentinel = "PC6_" + "DEFINITELY_NOT_A_REAL" + "_ENV_NAME"
    assert sentinel not in _referenced_names({sentinel})


def test_a_name_read_through_a_constant_is_seen() -> None:
    """`_X_ENV = "NAME"` then `os.getenv(_X_ENV)` — the indirect read.

    `GROBID_URL` is exactly this shape in
    `references/grobid_reference_service.py`, and before this pattern the guard
    cleared it only because an unrelated test happens to spell the name
    literally. Deleting or refactoring that test would have reported a live
    production knob as dead — the false-positive direction this file calls the
    worse failure.
    """
    assert "GROBID_URL" in _referenced_names({"GROBID_URL"})


def test_setting_a_variable_is_not_reading_it() -> None:
    """`monkeypatch.setenv("X", …)` must not count as a consumer.

    The env-shaped-reader pattern is deliberately loose about the function name,
    which made `setenv`/`delenv` match. No compose name relies on that today; the
    exclusion is so none quietly starts to, and it is the same coincidental-
    collision class the read-context rewrite was built to close.
    """
    import re

    for pattern in _READ_PATTERNS:
        regex = re.compile(pattern.replace("{name}", "SOME_NAME"))
        assert not regex.search('monkeypatch.setenv("SOME_NAME", "x")'), pattern
        assert not regex.search('monkeypatch.delenv("SOME_NAME")'), pattern
    # And the control: a real read still matches.
    assert any(
        re.compile(p.replace("{name}", "SOME_NAME")).search('os.getenv("SOME_NAME")')
        for p in _READ_PATTERNS
    )
