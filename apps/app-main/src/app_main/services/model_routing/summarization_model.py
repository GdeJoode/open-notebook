"""Failover-wrapping LanguageModel adapter for summarization (Track J.4).

The summarization strategies (naive / treekg / raptor) call a single object's
``achat_complete(messages)`` and read ``response.content``. They were hard-wired
to construct an esperanto ``LanguageModel`` from the env-driven ``LLMConfig``
(always ollama). J.4 unifies summarization with the other LLM stages by letting
the app inject a ``language_model`` into :class:`SummarizationConfig`.

:class:`FailoverSummarizationModel` is that injected object. It looks like an
esperanto ``LanguageModel`` to the strategies (``achat_complete`` -> object with
``.content``) but internally resolves the SUMMARIZATION route under the run's
privacy mode and runs each call through the J.2 failover executor with the J.4
error mapping — so a cloud summarization failure transparently fails over to the
next provider and ultimately to local Ollama (CLOUD mode), and a PRIVATE run
never constructs a cloud model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from app_main.services.model_routing.route_resolver import LLMTask, PrivacyMode


@dataclass
class _Response:
    """Minimal esperanto-``ChatCompletion``-shaped response.

    Strategies read ``.content``; we expose only that. Kept tiny so the adapter
    is a faithful drop-in for the single attribute the summarization path uses.
    """

    content: str


class FailoverSummarizationModel:
    """Drop-in ``LanguageModel`` whose ``achat_complete`` runs the failover route.

    Built once per summarization run with the resolved privacy mode + ids; each
    ``achat_complete`` resolves the SUMMARIZATION route fresh (so breaker state +
    key availability are honored) and executes it with per-provider failover.
    """

    def __init__(
        self,
        *,
        mode: PrivacyMode,
        model_id: Optional[str] = None,
        source_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
    ) -> None:
        self._mode = mode
        self._model_id = model_id
        self._source_id = source_id
        self._notebook_id = notebook_id
        # Last-served provenance (read by the service for telemetry/logging).
        self.served_provider: Optional[str] = None
        self.served_model_id: Optional[str] = model_id
        self.was_failover: bool = False
        self.fallback_from: List[str] = []

    async def achat_complete(self, messages: List[dict], **_kwargs: Any) -> _Response:
        """Run one routed chat completion across the SUMMARIZATION route.

        ``messages`` is the esperanto-shaped ``[{role, content}, ...]`` list the
        strategies build; we split it into system/user for ``call_candidate``.
        """
        from functools import partial

        from app_main.dependencies import get_failover_executor, get_route_resolver
        from app_main.services.model_routing.error_mapping import (
            is_failover_eligible,
        )
        from app_main.services.model_routing.llm_call import call_candidate
        from app_main.services.model_routing.telemetry import record_routing_event

        system = ""
        user = ""
        for m in messages:
            role = m.get("role")
            if role == "system":
                system = m.get("content", "")
            elif role == "user":
                # Last user message wins (strategies send a single user turn).
                user = m.get("content", "")
            else:
                user = m.get("content", user)

        resolver = get_route_resolver()
        route = await resolver.resolve(
            LLMTask.SUMMARIZATION, self._mode, model_id=self._model_id
        )
        if not route.ordered_candidates:
            raise RuntimeError(
                "No model candidates resolved for SUMMARIZATION "
                f"(mode={self._mode})."
            )

        executor = get_failover_executor()
        executor._is_failover_eligible = is_failover_eligible  # noqa: SLF001

        async def _call(system_prompt, user_prompt, candidate):
            return await call_candidate(candidate, system_prompt, user_prompt)

        result = await executor.execute_with_failover(
            route, partial(_call, system, user)
        )

        self.served_provider = result.served_provider
        self.served_model_id = result.served_model_id
        self.was_failover = result.was_failover
        self.fallback_from = result.fallback_from

        await record_routing_event(
            task=LLMTask.SUMMARIZATION,
            served_provider=result.served_provider,
            served_model_id=result.served_model_id,
            was_failover=result.was_failover,
            fallback_from=result.fallback_from,
            source_id=self._source_id,
            notebook_id=self._notebook_id,
        )

        return _Response(content=result.value.text)
