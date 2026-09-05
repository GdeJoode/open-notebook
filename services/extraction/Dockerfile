FROM python:3.12-slim-bookworm AS base

WORKDIR /app

COPY services/extraction/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ontology YAML files
COPY packages/ontology-manager/ontologies /app/ontologies

# Copy shared modules (model routing) — single canonical source
COPY packages/shared/src/shared/model_routing.py /app/model_routing.py
COPY packages/shared/src/shared/model_routing.yaml /app/model_routing.yaml
# PC.6: the coherence checks travel with the routing they check.
COPY packages/shared/src/shared/config_coherence.py /app/config_coherence.py

# PC.3: the identity rule, so this service derives `name_key` with the SAME
# function `EntityRepository` uses — two writers deriving identity
# independently is how one name becomes two rows.
#
# The four modules are copied WITHOUT any `__init__.py`, deliberately.
# `shared/__init__.py` imports the whole package and pulls in `packaging`,
# `tiktoken` and `surrealdb_service`, none of which belong in this image —
# it crash-looped the container when tried. Omitting the `__init__` files
# makes `shared` a namespace package, so only these four load. Their own
# imports are `re`, `json`, `os`, `pathlib` and `loguru`, which is already a
# dependency here.
COPY packages/shared/src/shared/utils/name_normalizer.py /app/shared/utils/name_normalizer.py
COPY packages/shared/src/shared/utils/nl_normalization.py /app/shared/utils/nl_normalization.py
COPY packages/shared/src/shared/utils/org_aliases.py /app/shared/utils/org_aliases.py
COPY packages/shared/src/shared/config/alias_overrides.py /app/shared/config/alias_overrides.py

# Copy the service API and modules
COPY services/extraction/api.py /app/api.py
COPY services/extraction/vocabulary_manager.py /app/vocabulary_manager.py
COPY services/extraction/entity_validator.py /app/entity_validator.py

# Create vocabularies directory (populated via POST /vocabularies/refresh)
RUN mkdir -p /app/vocabularies

EXPOSE 8103

# LLM (Ollama on host)
ENV OLLAMA_URL="http://host.docker.internal:11434"
# EXTRACTION_MODEL / EXTRACTION_NUM_CTX removed (PC.6): nothing consumed them.
# Extraction routes through packages/shared/src/shared/model_routing.yaml.
ENV EXTRACTION_CHUNK_SIZE="4000"

# Embedding
ENV EMBEDDING_MODEL="mxbai-embed-large"

# SurrealDB
ENV SURREALDB_URL="ws://host.docker.internal:8000/rpc"
ENV SURREALDB_NS="open_notebook"
ENV SURREALDB_DB="open_notebook"
ENV SURREALDB_USER="root"
ENV SURREALDB_PASS="root"

# Ontology
ENV ONTOLOGY_DIR="/app/ontologies"
ENV DEFAULT_ONTOLOGY="regiodeal"

# Model routing
ENV DEFAULT_PRIVACY="internal"
ENV NVIDIA_API_KEY=""
ENV NVIDIA_API_URL="https://integrate.api.nvidia.com/v1"

# Vocabularies
ENV VOCAB_DIR="/app/vocabularies"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8103"]
