FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS base

# Python 3.12 + system deps (ffmpeg for video audio extraction, ffprobe for metadata)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    gcc g++ \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies inside a venv (PEP 668 compliance)
# ---------------------------------------------------------------------------

RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY services/whisperx/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Copy ingestion pipeline source (audio, models, config, exporters)
# ---------------------------------------------------------------------------

COPY pipelines/ingestion/src/ingestion /app/ingestion

# Copy the service API
COPY services/whisperx/api.py /app/api.py

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

EXPOSE 8102

# NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# WhisperX defaults — large-v3 model, CUDA, full features
ENV WHISPERX_MODEL="large-v3"
ENV WHISPERX_LANGUAGE=""
ENV WHISPERX_DEVICE="cuda"
ENV WHISPERX_COMPUTE_TYPE="float16"
ENV WHISPERX_BATCH_SIZE="16"
ENV WHISPERX_DIARIZATION="true"
ENV WHISPERX_MIN_SPEAKERS=""
ENV WHISPERX_MAX_SPEAKERS=""
ENV WHISPERX_WORD_TIMESTAMPS="true"

# HuggingFace token for pyannote diarization (required for speaker detection)
ENV HF_TOKEN=""

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8102"]
