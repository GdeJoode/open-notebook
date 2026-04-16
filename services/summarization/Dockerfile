FROM python:3.12-slim-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------

COPY services/summarization/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Copy summarization pipeline source
# ---------------------------------------------------------------------------

COPY pipelines/summarization/src/summarization /app/summarization

# Copy the service API, presets module, and preset YAML files
COPY services/summarization/api.py /app/api.py
COPY services/summarization/presets.py /app/presets.py
COPY services/summarization/presets /app/presets

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

EXPOSE 8101

# LLM defaults (Ollama on host)
ENV SUMMARIZATION_PROVIDER="ollama"
ENV SUMMARIZATION_MODEL="llama3.1:8b-instruct-q4_0"

ENV SUMMARIZATION_BASE_URL="http://host.docker.internal:11434"
ENV SUMMARIZATION_TEMPERATURE="0.1"
ENV SUMMARIZATION_MAX_TOKENS="3000"

# Embedding defaults (Ollama on host)
ENV EMBEDDING_PROVIDER="ollama"
ENV EMBEDDING_MODEL="nomic-embed-text"
ENV EMBEDDING_BASE_URL="http://host.docker.internal:11434"

# Optional API key for cloud providers
ENV LLM_API_KEY=""

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8101"]
