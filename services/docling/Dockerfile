FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS base

# Python 3.12 + system deps required by docling, easyocr, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    gcc g++ \
    libgl1 libglib2.0-0 \
    tesseract-ocr \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies inside a venv (PEP 668 compliance)
# ---------------------------------------------------------------------------

RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY services/docling/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Copy ingestion pipeline source (models, config, parsers, exporters only)
# ---------------------------------------------------------------------------

COPY pipelines/ingestion/src/ingestion /app/ingestion

# Copy the service API and post-processing modules
COPY services/docling/api.py /app/api.py
COPY services/docling/pymupdf_enricher.py /app/pymupdf_enricher.py
COPY services/docling/annotation_extractor.py /app/annotation_extractor.py
COPY services/docling/handwriting_detector.py /app/handwriting_detector.py
COPY services/docling/vlm_reader.py /app/vlm_reader.py

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

EXPOSE 8100

# Preload models on startup to avoid first-request latency
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DOCLING_DEVICE="cuda"
ENV DOCLING_NUM_THREADS="4"
ENV DOCLING_DO_OCR="true"
ENV DOCLING_OCR_ENGINE="easyocr"
ENV DOCLING_OCR_LANG="en"
ENV DOCLING_DO_TABLE_STRUCTURE="true"
ENV DOCLING_TABLE_MODE="accurate"
ENV DOCLING_GENERATE_PAGE_IMAGES="false"
ENV DOCLING_GENERATE_PICTURE_IMAGES="true"
ENV DOCLING_IMAGES_SCALE="2.0"
ENV DOCLING_DO_PICTURE_CLASSIFICATION="true"
ENV DOCLING_DO_PICTURE_DESCRIPTION="true"
ENV DOCLING_VLM_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"
ENV DOCLING_DO_FORMULA_EXTRACTION="true"
ENV DOCLING_DO_CODE_ENRICHMENT="true"
ENV DOCLING_ORGANIZE_BY_PAGE="true"

# VLM for handwriting recognition (Ollama on host)
ENV VLM_PROVIDER="ollama"
ENV VLM_MODEL="granite3.2-vision"
ENV VLM_BASE_URL="http://host.docker.internal:11434"
ENV VLM_TIMEOUT="120"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8100"]
