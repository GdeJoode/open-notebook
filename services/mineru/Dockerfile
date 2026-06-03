# MinerU document-parsing service.
#
# Standalone GPU-backed FastAPI service running MinerU 2.x. Mirrors the
# layout of services/docling so the two parsers can be swapped behind
# the parser_engine routing introduced in Phase A.1b.
#
# Parser-selection inspiration credit: the dual-parser pattern (docling
# alongside MinerU with confidence-driven fallback) follows the design
# discussed in SenolIsci/mykg (MIT). This service is an independent
# implementation; only the architectural pattern is borrowed.

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS base

# Python 3.12 + system deps required by MinerU (PyTorch + pillow + libmagic
# + libGL for OpenCV transitively pulled in by mineru[all]).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    gcc g++ \
    libgl1 libglib2.0-0 libmagic1 \
    poppler-utils \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies inside a venv (PEP 668 compliance)
# ---------------------------------------------------------------------------

RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY services/mineru/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Copy the service API
# ---------------------------------------------------------------------------

COPY services/mineru/api.py /app/api.py

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

EXPOSE 8104

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# MinerU runtime configuration. Defaults match what the docling service does:
# CUDA when available, formula + table extraction on, OCR auto-detected,
# English language hint for OCR.
ENV MINERU_DEVICE="cuda"
ENV MINERU_BACKEND="pipeline"
ENV MINERU_METHOD="auto"
ENV MINERU_LANG="en"
ENV MINERU_FORMULA_ENABLE="true"
ENV MINERU_TABLE_ENABLE="true"
# Use modelscope mirror when huggingface is unreachable; default to HF.
# ENV MINERU_MODEL_SOURCE="modelscope"

# Persistent model cache so first-run downloads (~5-8 GB) survive container
# restarts. Mount a named volume here from docker-compose.
ENV HF_HOME="/data/models/huggingface"
ENV MODELSCOPE_CACHE="/data/models/modelscope"
ENV TRANSFORMERS_CACHE="/data/models/huggingface"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8104"]
