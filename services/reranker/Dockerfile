FROM python:3.12-slim AS base

# System deps: build tools for any wheels that need compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies inside a venv (PEP 668 compliance)
# ---------------------------------------------------------------------------

RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY services/reranker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the service API
COPY services/reranker/api.py /app/api.py

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

EXPOSE 8105

# Multilingual cross-encoder (corpus is Dutch). Override to tune size/latency.
ENV RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
# CPU by default; set RERANKER_DEVICE=cuda + add an nvidia reservation to use GPU.
ENV RERANKER_DEVICE=""
# Cache HuggingFace model downloads under the mounted volume so the ~2 GB
# model is fetched once across container restarts.
ENV HF_HOME="/data/models"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8105"]
