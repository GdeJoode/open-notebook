"""Modelaanroepen uit app-main: de dunne aanroeper van ``ModelGate`` (ADR 0006).

:mod:`.llm_call` stuurt elke aanroep via :class:`llm_manager.gate.ModelGate` naar
llama-swap; :mod:`.telemetry` schrijft het auditlog ``model_call`` (ADR 0007).
De Track J-routering (providerketens, failover, privacymodus) is verwijderd in
F2-09.
"""

from app_main.services.model_routing.llm_call import LLMTask

__all__ = ["LLMTask"]
