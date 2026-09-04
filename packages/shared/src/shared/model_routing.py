"""
Model routing — selects LLM provider and model based on privacy level.

Reads model_routing.yaml for defaults. Each API request can override
the privacy level or specify a specific model.

Usage:
    from model_routing import get_model_config, call_llm, PrivacyLevel

    # Get config for a pipeline step
    config = get_model_config("extraction", privacy="public")
    # → {"provider": "nvidia", "model": "mistralai/mistral-large-3-675b-instruct-2512", ...}

    # Call LLM with automatic routing
    response = await call_llm(
        messages=[{"role": "user", "content": "Extract entities..."}],
        step="extraction",
        privacy="internal",
    )

    # Override with specific model
    response = await call_llm(
        messages=[...],
        step="extraction",
        model_override={"provider": "nvidia", "model": "...", "api_key": "..."},
    )

Works in Docker services and main app — reads config from YAML + env vars.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Privacy levels
# ---------------------------------------------------------------------------

class ProviderUnavailableError(RuntimeError):
    """A route names a provider that is declared unavailable.

    Track PC.6. Raised at RESOLUTION rather than at call time, because that is
    the last point where the reason is still known — one step later the failure
    is an HTTP error from a vendor, which cannot say that the route was retired
    on purpose. The message names the step, the privacy level and the reason, so
    the fix is readable from the exception alone.
    """

    def __init__(
        self, *, provider: str, step: str, privacy: str, model: str, reason: str
    ) -> None:
        self.provider, self.step, self.privacy = provider, step, privacy
        self.model, self.reason = model, reason
        super().__init__(
            f"routing {step!r}/{privacy!r} to provider {provider!r} "
            f"(model {model!r}), which is declared unavailable: {reason} "
            f"— set providers.{provider}.available: true in model_routing.yaml "
            f"to restore it, or route this step elsewhere."
        )


PRIVACY_LEVELS = ("public", "internal", "confidential")
DEFAULT_PRIVACY = os.getenv("DEFAULT_PRIVACY", "internal")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_config: Optional[Dict] = None
_config_path = Path(__file__).parent / "model_routing.yaml"


def _resolve_env(value: str) -> str:
    """Resolve ${VAR:-default} patterns in a string."""
    def replacer(match):
        var = match.group(1)
        default = match.group(3) or ""
        return os.getenv(var, default)
    return re.sub(r"\$\{(\w+)(:-([^}]*))?\}", replacer, str(value))


def _load_config() -> Dict:
    """Load model routing config from YAML."""
    global _config
    if _config is not None:
        return _config

    # Try multiple paths (Docker service vs main app vs development)
    paths = [
        _config_path,
        Path("/app/model_routing.yaml"),
        Path(os.getenv("MODEL_ROUTING_CONFIG", "")),
    ]

    for p in paths:
        if p and p.exists():
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            # Resolve environment variables in provider configs
            providers = raw.get("providers", {})
            for pname, pconfig in providers.items():
                for key, val in pconfig.items():
                    if isinstance(val, str):
                        pconfig[key] = _resolve_env(val)
            _config = raw
            logger.info(f"Model routing config loaded from {p}")
            return _config

    logger.warning("No model_routing.yaml found, using fallback defaults")
    _config = {"routing": {}, "providers": {}, "defaults": {"default_privacy": "internal"}}
    return _config


def reload_config():
    """Force reload of config (e.g. after editing YAML)."""
    global _config
    _config = None
    return _load_config()


# ---------------------------------------------------------------------------
# Model config resolution
# ---------------------------------------------------------------------------


def get_model_config(
    step: str,
    privacy: Optional[str] = None,
    model_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get the model configuration for a pipeline step.

    Resolution order:
    1. model_override (explicit per-request) — highest priority
    2. routing config for step + privacy level
    3. routing config for step + "any" (privacy-independent)
    4. fallback to local Ollama

    Args:
        step: Pipeline step ("extraction", "summarization", "embedding", etc.)
        privacy: Privacy level ("public", "internal", "confidential").
            None = use default from config.
        model_override: Explicit override dict with provider/model/api_key.

    Returns:
        Dict with: provider, model, base_url, api_key, temperature, max_tokens, etc.
    """
    config = _load_config()

    # Explicit override takes priority
    if model_override and model_override.get("provider"):
        resolved = dict(model_override)
        # Fill in base_url from provider config if not specified
        if "base_url" not in resolved:
            provider_name = resolved["provider"]
            provider_conf = config.get("providers", {}).get(provider_name, {})
            resolved["base_url"] = provider_conf.get("base_url", "")
        if "api_key" not in resolved:
            provider_name = resolved["provider"]
            provider_conf = config.get("providers", {}).get(provider_name, {})
            resolved["api_key"] = provider_conf.get("api_key", "")
        return resolved

    # Determine privacy level
    privacy = privacy or config.get("defaults", {}).get("default_privacy", DEFAULT_PRIVACY)
    if privacy not in PRIVACY_LEVELS:
        privacy = "internal"

    # Look up routing
    routing = config.get("routing", {}).get(step, {})

    # Try privacy-specific route, then "any"
    route = routing.get(privacy) or routing.get("any")
    if not route:
        # Fallback: local Ollama
        logger.warning(f"No routing for step='{step}' privacy='{privacy}', falling back to Ollama")
        route = {
            "provider": "ollama",
            "model": "llama3.1:8b-instruct-q4_0",
            "temperature": 0.1,
            "max_tokens": 2048,
        }

    # Resolve provider details
    provider_name = route.get("provider", "ollama")
    provider_conf = config.get("providers", {}).get(provider_name, {})

    # PC.6: a provider declared unavailable raises here rather than being tried.
    # Resolution is the last point where the reason is still known — one step
    # later this is an HTTP 401 from a vendor, which says nothing about the fact
    # that the route was deliberately retired. The routes themselves are kept:
    # deleting them would lose what the cloud path was.
    if provider_conf.get("available") is False:
        raise ProviderUnavailableError(
            provider=provider_name,
            step=step,
            privacy=privacy,
            model=route.get("model", "?"),
            reason=str(provider_conf.get("reason", "")).strip()
            or "declared unavailable in model_routing.yaml",
        )

    resolved = dict(route)
    resolved["base_url"] = provider_conf.get("base_url", os.getenv("OLLAMA_URL", "http://localhost:11434"))
    resolved["api_key"] = provider_conf.get("api_key", "")
    resolved["provider_type"] = provider_conf.get("type", "ollama")

    return resolved


# ---------------------------------------------------------------------------
# LLM call — unified interface for all providers
# ---------------------------------------------------------------------------


async def call_llm(
    messages: List[Dict[str, str]],
    step: str = "extraction",
    privacy: Optional[str] = None,
    model_override: Optional[Dict[str, Any]] = None,
    stream: bool = False,
) -> str:
    """Call an LLM with automatic provider routing.

    Handles Ollama and OpenAI-compatible APIs (NVIDIA, Anthropic, OpenAI)
    through a unified interface.

    Args:
        messages: Chat messages [{"role": "user", "content": "..."}]
        step: Pipeline step for routing.
        privacy: Privacy level.
        model_override: Explicit model override.
        stream: Whether to stream (only returns full text, not chunks).

    Returns:
        The LLM response text.
    """
    config = get_model_config(step, privacy, model_override)

    provider_type = config.get("provider_type", "ollama")
    base_url = config.get("base_url", "")
    model = config.get("model", "")
    api_key = config.get("api_key", "")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 2048)

    logger.debug(f"LLM call: step={step} privacy={privacy} provider={config.get('provider')} model={model}")

    if provider_type == "ollama":
        return await _call_ollama(base_url, model, messages, temperature, max_tokens, config)
    elif provider_type == "openai_compatible":
        return await _call_openai_compatible(base_url, model, api_key, messages, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


async def _call_ollama(
    base_url: str,
    model: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    config: Dict,
) -> str:
    """Call Ollama API."""
    # Build prompt from messages
    system = ""
    user = ""
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            user = msg["content"]

    prompt = f"{system}\n\n{user}" if system else user

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": config.get("num_ctx", 8192),
                },
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


async def _call_openai_compatible(
    base_url: str,
    model: str,
    api_key: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call OpenAI-compatible API (NVIDIA, OpenAI, Anthropic via proxy)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""


# ---------------------------------------------------------------------------
# Embedding — always local
# ---------------------------------------------------------------------------


async def embed_text(
    text: str,
    step: str = "embedding",
    privacy: Optional[str] = None,
) -> List[float]:
    """Generate embedding using the configured model (always local)."""
    config = get_model_config(step, privacy)
    base_url = config.get("base_url", "")
    model = config.get("model", "mxbai-embed-large")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embs = data.get("embeddings", [[]])
        return embs[0] if embs else []


# ---------------------------------------------------------------------------
# Convenience: get routing summary
# ---------------------------------------------------------------------------


def get_routing_summary() -> Dict[str, Any]:
    """Get a summary of current routing configuration."""
    config = _load_config()
    routing = config.get("routing", {})

    summary = {}
    for step, routes in routing.items():
        summary[step] = {}
        for privacy, route in routes.items():
            summary[step][privacy] = f"{route.get('provider', '?')}:{route.get('model', '?')}"

    return {
        "default_privacy": config.get("defaults", {}).get("default_privacy", DEFAULT_PRIVACY),
        "providers": list(config.get("providers", {}).keys()),
        "routing": summary,
    }
