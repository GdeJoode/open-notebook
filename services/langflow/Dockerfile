# Langflow voor het signaalsysteem (F2-12; ADR 0017, ADR 0022).
#
# Vaste versie 1.13.0 met digest (P2). Daarbovenop:
# - llm_manager (modelregister en ModelGate, F2-07) uit packages/llm-manager,
#   via de extra buildcontext `llm_manager` (Compose: additional_contexts);
#   llm_manager gebruikt alleen pydantic, pyyaml, loguru en httpx, die het image
#   al heeft (geen pip install, niets van internet bij het bouwen);
# - het startscript en de scripts voor laden en exporteren van flows.
# De eigen componenten, de componentenlijst en het modelregister komen uit de
# kloon van HEAD (/app/flows-git, alleen-lezen); ze zitten niet in het image.
FROM langflowai/langflow:1.13.0@sha256:ff2716e017f4a4bbb9e83556bf66c3d5d70a1d3eb7cffd893312600ba84e303f

USER root
COPY --from=llm_manager . /app/signaal/lib/llm_manager/
COPY scripts/start.sh scripts/flows_laden.py scripts/flows_exporteren.py scripts/langflow_api.py /app/signaal/
RUN rm -rf /app/signaal/lib/llm_manager/__pycache__ /app/signaal/lib/llm_manager/providers/__pycache__ \
 && echo /app/signaal/lib > /app/.venv/lib/python3.14/site-packages/signaal.pth \
 && chmod -R a+rX,go-w /app/signaal && chmod 755 /app/signaal/start.sh \
 && mkdir -p /app/data && chown 1000:0 /app/data

# Agent- en MCP-interfaces van Langflow uit (FO 11, ADR 0017, ADR 0022).
# In 1.13.0 hangt langflow/api/router.py ze altijd in, ook met
# LANGFLOW_MCP_SERVER_ENABLED=false: aangemeld gaf /api/v1/mcp/sse HTTP 200, en
# /api/v1/agentic/mcp is een tweede MCP-server (F2-12, review B1). Weg gaan: de
# MCP-server (v1, per project, v2-configuratie), de MCP-server van de assistent,
# A2A (/api/v1/a2a) en de OpenAI-Responses-API (/api/v1/responses). Ook de
# Store (/api/v1/store): met LANGFLOW_STORE=false vroegen de eindpunten nog
# api.langflow.store op (controle L7). Na de herreview ook de spraak-websocket
# (/api/v1/voice) en de hele assistent (_include_agentic_router: assist,
# execute, run). main.py hangt de MCP-router alleen in als de instelling aan
# staat. Ontbreekt een regel (andere versie), of blijft daarna in langflow/api
# een include_router met mcp, a2a, openai_responses, store_router of
# voice_mode_router over, of de aanroep van _include_agentic_router, dan faalt
# de build.
RUN d=/app/.venv/lib/python3.14/site-packages/langflow/api \
 && for r in 'router_v1\.include_router\(mcp_router\)' 'router_v1\.include_router\(mcp_projects_router\)' \
             'router_v2\.include_router\(mcp_router_v2\)' 'router_v1\.include_router\(agentic_mcp_router\)' \
             'router_v1\.include_router\(a2a_router\)' 'router_v1\.include_router\(openai_responses_router\)' \
             'router_v1\.include_router\(store_router\)' 'router_v1\.include_router\(voice_mode_router\)' \
             '_include_agentic_router\(\)'; do \
      grep -qE "^[[:space:]]*$r\$" "$d/router.py" || { echo "niet gevonden in router.py: $r" >&2; exit 1; }; \
      sed -i -E "/^[[:space:]]*$r\$/d" "$d/router.py"; done \
 && if grep -rnE 'include_router\(.*(mcp|a2a|openai_responses|store_router|voice_mode_router)' "$d" --include=*.py \
      || grep -nE '^[[:space:]]*_include_agentic_router\(\)' "$d/router.py"; then \
      echo "nog een MCP- of agentrouter in langflow/api" >&2; exit 1; fi \
 && rm -f "$d"/__pycache__/router.*.pyc

# Geen versiecontrole bij het starten (F2-12, controle L7): `langflow run` vraagt
# in print_banner altijd de nieuwste versie op bij pypi.org; de telling vond
# daarvoor DNS-vragen. fetch_latest_version geeft nu direct None.
RUN f=/app/.venv/lib/python3.14/site-packages/langflow/utils/version.py \
 && d='def fetch_latest_version(package_name: str, *, include_prerelease: bool) -> str | None:' \
 && grep -qxF "$d" "$f" || { echo "niet gevonden in $f: $d" >&2; exit 1; } \
 && sed -i "/^def fetch_latest_version(/a\\    return None  # signaalsysteem F2-12: geen versiecontrole bij pypi.org" "$f" \
 && grep -qF 'return None  # signaalsysteem F2-12' "$f" \
 && rm -f /app/.venv/lib/python3.14/site-packages/langflow/utils/__pycache__/version.*.pyc

# Verplicht in dit image: het standaardpad van llm_manager wijst naar een
# checkout van de repository (review F2-07). Het register, de componentenlijst
# en de eigen componenten komen uit de kloon van HEAD (start.sh), net als de
# flows (review F2-12, punt 3; herreview punt 4).
ENV SIGNAAL_MODELREGISTER=/app/flows-git/config/modelregister.yaml \
    SIGNAAL_COMPONENTENLIJST=/app/flows-git/config/langflow-componenten.yaml

USER 1000
ENTRYPOINT ["/app/signaal/start.sh"]
CMD []
