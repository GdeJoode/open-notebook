"""
LLM Manager: de modelpoort, het modelregister en usage tracking.

Elke modelaanroep loopt via :mod:`llm_manager.aanroep` en
:class:`llm_manager.gate.ModelGate` naar llama-swap (ADR 0006). Welke modellen
er zijn, staat in het register (:mod:`llm_manager.register`, ADR 0007).
"""

from llm_manager.usage import UsageMetrics, UsageTracker

__all__ = [
    "UsageTracker",
    "UsageMetrics",
]
