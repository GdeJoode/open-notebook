FROM python:3.12-slim-bookworm AS base

WORKDIR /app

COPY services/extraction/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ontology YAML files
COPY packages/ontology-manager/ontologies /app/ontologies

# Copy shared modules (model routing)
COPY services/shared/model_routing.py /app/model_routing.py
COPY services/shared/model_routing.yaml /app/model_routing.yaml

# Copy the service API and modules
COPY services/extraction/api.py /app/api.py
COPY services/extraction/vocabulary_manager.py /app/vocabulary_manager.py
COPY services/extraction/entity_validator.py /app/entity_validator.py

# Create vocabularies directory (populated via POST /vocabularies/refresh)
RUN mkdir -p /app/vocabularies

EXPOSE 8103

# LLM (Ollama on host)
ENV OLLAMA_URL="http://host.docker.internal:11434"
ENV EXTRACTION_MODEL="llama3.1:8b-instruct-q4_0"
ENV EXTRACTION_NUM_CTX="8192"
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
