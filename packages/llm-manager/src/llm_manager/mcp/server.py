"""MCP server for llm-manager access."""

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from llm_manager.manager import ModelManager, get_model_manager
from llm_manager.providers.registry import (
    get_provider_config as _get_provider_config,
    list_providers as _list_providers,
)
from llm_manager.usage.tracker import UsageTracker


def _get_tracker() -> UsageTracker:
    """Get the global usage tracker (same singleton as the REST API)."""
    from llm_manager.api.routers.usage import get_tracker

    return get_tracker()


def create_server() -> FastMCP:
    """Create and configure the llm-manager MCP server."""
    mcp = FastMCP(
        name="llm-manager",
        instructions="LLM provider management and usage tracking for open-notebook.",
    )

    @mcp.tool()
    async def list_providers() -> str:
        """List all LLM providers with configuration status and capabilities."""
        providers = _list_providers()
        result = []
        for p in providers:
            result.append({
                "name": p.name,
                "display_name": p.display_name,
                "is_configured": bool(os.environ.get(p.api_key_env_var)) or p.is_local,
                "is_local": p.is_local,
                "supports": {
                    "language": p.supports_language,
                    "embedding": p.supports_embedding,
                    "speech_to_text": p.supports_speech_to_text,
                    "text_to_speech": p.supports_text_to_speech,
                    "vision": p.supports_vision,
                    "tools": p.supports_tools,
                },
            })
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def get_provider_details(provider_name: str) -> str:
        """Get detailed info for a specific provider including config and base URL."""
        config = _get_provider_config(provider_name)
        if not config:
            return json.dumps({"error": f"Provider '{provider_name}' not found"})
        data = config.model_dump()
        data["is_configured"] = bool(os.environ.get(config.api_key_env_var)) or config.is_local
        if config.base_url_env_var:
            data["active_base_url"] = os.environ.get(
                config.base_url_env_var, config.default_base_url
            )
        return json.dumps(data, indent=2)

    @mcp.tool()
    async def get_model_defaults() -> str:
        """Get current default model assignments (chat, embedding, TTS, STT, etc.)."""
        manager = get_model_manager()
        defaults = manager.get_defaults()
        if defaults is None:
            return json.dumps({"status": "No defaults configured yet"})
        return json.dumps(defaults.model_dump(), default=str, indent=2)

    @mcp.tool()
    async def get_usage_metrics() -> str:
        """Get aggregated usage metrics: total tokens, cost, requests by model/provider."""
        tracker = _get_tracker()
        metrics = tracker.get_metrics()
        return json.dumps(asdict(metrics), indent=2)

    @mcp.tool()
    async def get_usage_history(
        limit: int = 50,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        operation_type: Optional[str] = None,
    ) -> str:
        """Get recent usage history with optional filtering by model, provider, or operation."""
        tracker = _get_tracker()
        records = tracker.get_history(
            limit=limit,
            model_id=model_id,
            provider=provider,
            operation_type=operation_type,
        )
        result = [
            {
                "model_id": r.model_id,
                "model_name": r.model_name,
                "provider": r.provider,
                "operation_type": r.operation_type,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": r.cost_usd,
                "duration_ms": r.duration_ms,
                "success": r.success,
                "error_message": r.error_message,
            }
            for r in records
        ]
        return json.dumps(result, default=str, indent=2)

    @mcp.tool()
    async def get_cache_stats() -> str:
        """Get model instance cache statistics (entries, validity, TTL)."""
        manager = get_model_manager()
        stats = manager.get_cache_stats()
        return json.dumps(stats, indent=2)

    @mcp.tool()
    async def estimate_cost(
        input_tokens: int,
        output_tokens: int,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
    ) -> str:
        """Estimate cost for given token counts and per-1K pricing."""
        tracker = _get_tracker()
        total = tracker.estimate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        result = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total, 6),
        }
        return json.dumps(result, indent=2)

    return mcp


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="LLM Manager MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8202)
    args = parser.parse_args()

    server = create_server()
    transport_kwargs: Dict[str, Any] = {}
    if args.transport != "stdio":
        transport_kwargs["port"] = args.port
    server.run(transport=args.transport, **transport_kwargs)
