"""
LLM Manager: de modelpoort en het modelregister.

Elke modelaanroep loopt via :mod:`llm_manager.aanroep` en
:class:`llm_manager.gate.ModelGate` naar llama-swap (ADR 0006). Welke modellen
er zijn, staat in het register (:mod:`llm_manager.register`, ADR 0007); het
auditlog is ``model_call`` (ADR 0007).
"""
