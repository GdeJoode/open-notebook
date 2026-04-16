FROM python:3.12-slim-bookworm AS base

WORKDIR /app

COPY services/extraction/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ontology YAML files
COPY packages/ontology-manager/ontologies /app/ontologies

# Copy the service API
COPY services/extraction/api.py /app/api.py

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

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8103"]
