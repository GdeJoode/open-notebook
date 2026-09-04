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
    "PYTHONUNBUFFERED": "read by the Python runtime itself",
    "PYTHONPATH": "read by the Python runtime itself",
    "TZ": "read by the C library",
    "PATH": "read by the shell",
    "NVIDIA_VISIBLE_DEVICES": "read by the NVIDIA container runtime",
    "NVIDIA_DRIVER_CAPABILITIES": "read by the NVIDIA container runtime",
    "CUDA_VISIBLE_DEVICES": "read by CUDA",
    "HF_HOME": "read by huggingface_hub",
    "HUGGING_FACE_HUB_TOKEN": "read by huggingface_hub",
    "TRANSFORMERS_CACHE": "read by transformers",
    "TORCH_HOME": "read by torch",
    "OMP_NUM_THREADS": "read by OpenMP",
    "SURREAL_PASSWORD": "read by the SurrealDB image entrypoint",
    "SURREAL_USER": "read by the SurrealDB image entrypoint",
    "SURREAL_EXPERIMENTAL_GRAPHQL": "read by the SurrealDB binary",
    "SURREAL_ROCKSDB_BLOCK_CACHE_SIZE": "read by the SurrealDB binary",
    # Verified read at `esperanto/utils/timeout.py:18` — a dependency, so outside
    # the tracked tree this scan walks. Checked rather than assumed: the compose
    # comment claims it raises the LLM timeout to 600s, and a claim like that is
    # exactly what this guard exists to make someone verify.
    "ESPERANTO_LLM_TIMEOUT": "read by esperanto (utils/timeout.py)",
}

_ENV_LINE = re.compile(r"^\s+-\s+([A-Z][A-Z0-9_]*)=", re.M)


def _compose_env_names() -> Set[str]:
    return set(_ENV_LINE.findall(COMPOSE.read_text(encoding="utf-8")))


def _referenced_names() -> Set[str]:
    """Every uppercase env-style token appearing in tracked source or config.

    Deliberately generous — a substring match over the tree. The question is
    "does anything mention this name at all", and a name that fails even that is
    unambiguously dead.
    """
    listed = subprocess.run(
        ["git", "ls-files", "--", "*.py", "*.yaml", "*.yml", "*.ts", "*.tsx", "*.sh"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    seen: Set[str] = set()
    for rel in listed:
        if rel == "docker-compose.yml":
            continue
        path = REPO_ROOT / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        seen.update(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
    return seen


def test_every_compose_env_name_is_referenced_somewhere() -> None:
    names = _compose_env_names()
    assert len(names) > 20, f"walker control: only {len(names)} env names parsed"

    referenced = _referenced_names()
    assert "OLLAMA_URL" in referenced, "detector control: a known-live name is missing"

    dead = sorted(n for n in names if n not in referenced and n not in _ALLOWED)
    assert not dead, (
        f"set in docker-compose.yml and referenced nowhere in the tree: {dead}. "
        f"An operator edits compose; a name that lives only there is a control "
        f"that configures nothing. Delete it, or wire it and say where."
    )


def test_the_detector_would_catch_a_planted_name() -> None:
    """Mutant control: the scan must be able to fail.

    Without it, the test above passes equally well against a `_referenced_names`
    that returns every possible token — which is what a generous substring match
    is one bug away from.
    """
    referenced = _referenced_names()
    assert "PC6_DEFINITELY_NOT_A_REAL_ENV_NAME" not in referenced
