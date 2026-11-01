# Build stage
FROM python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS builder

# Install uv using the official method
COPY --from=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563 /uv /uvx /bin/

# Install system dependencies required for building certain Python packages.
# Geen apt-get upgrade: de basis ligt vast op digest (F2-10).
RUN apt-get update && apt-get install -y \
    gcc g++ git make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Node.js met vaste versie en SHA-256 (uit nodejs.org/dist/v20.19.5/SHASUMS256.txt),
# in plaats van het NodeSource-script (F2-10). De runtime neemt /opt/node over.
ARG NODE_VERSION=20.19.5
ARG NODE_SHA256=4eba5fbe1fb10753bc06e42f001a91c5cec16798b7764a3e9257adc59af47fe1
RUN curl -fsSLo /tmp/node.tar.gz "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" \
    && echo "${NODE_SHA256}  /tmp/node.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/node && tar -xzf /tmp/node.tar.gz -C /opt/node --strip-components=1 \
    && rm /tmp/node.tar.gz
ENV PATH="/opt/node/bin:${PATH}"

# Set build optimization environment variables
ENV MAKEFLAGS="-j$(nproc)"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Set the working directory in the container to /app
WORKDIR /app

# ---------------------------------------------------------------------------
# Dependency layer — cached unless a manifest or the lockfile changes.
# Copy ONLY the dependency-defining files first so an app-code change does not
# bust the (multi-GB, CUDA/torch) external-dependency install. This is a uv
# workspace, so every member needs its pyproject.toml present for the resolve;
# the member SOURCE is copied in the next layer. README.md is needed for the
# root package's hatchling metadata.
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock README.md ./
COPY apps/app-main/pyproject.toml apps/app-main/
COPY packages/file-manager/pyproject.toml packages/file-manager/
COPY packages/job-queue/pyproject.toml packages/job-queue/
COPY packages/llm-manager/pyproject.toml packages/llm-manager/
COPY packages/ontology-manager/pyproject.toml packages/ontology-manager/
COPY packages/semantic-intelligence/pyproject.toml packages/semantic-intelligence/
COPY packages/shared/pyproject.toml packages/shared/
COPY packages/surrealdb-service/pyproject.toml packages/surrealdb-service/
COPY packages/zotero-integration/pyproject.toml packages/zotero-integration/
COPY pipelines/embeddings/pyproject.toml pipelines/embeddings/
COPY pipelines/entity-filtering/pyproject.toml pipelines/entity-filtering/
COPY pipelines/ingestion/pyproject.toml pipelines/ingestion/
COPY pipelines/ontology-extraction/pyproject.toml pipelines/ontology-extraction/
COPY pipelines/retrieval/pyproject.toml pipelines/retrieval/
COPY pipelines/summarization/pyproject.toml pipelines/summarization/

# Install third-party dependencies only (NOT the workspace packages), with a
# BuildKit cache mount so wheels survive across builds even if this layer reruns.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# ---------------------------------------------------------------------------
# Source layer — reruns on any code change, but fast: external deps are already
# installed, so this only adds the editable workspace packages.
# ---------------------------------------------------------------------------
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-packages

# Install frontend dependencies and build (npm cache mounted so `npm ci` is fast)
WORKDIR /app/frontend
RUN --mount=type=cache,target=/root/.npm \
    npm ci
RUN npm run build

# Het tiktoken-encodingbestand bij de build ophalen: in de container is geen
# internet (F2-10). Zelfde pad in de runtime via TIKTOKEN_CACHE_DIR.
RUN TIKTOKEN_CACHE_DIR=/opt/tiktoken /app/.venv/bin/python -c \
    "import tiktoken; tiktoken.get_encoding('o200k_base')"

# Return to app root
WORKDIR /app

# Runtime stage
FROM python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS runtime

# Install only runtime system dependencies (no build tools); geen apt-get upgrade.
# git: het flowregister leest een alleen-lezen kloon van de flowmap (F2-04, F2-10)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    supervisor \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/node /opt/node
ENV PATH="/opt/node/bin:${PATH}"
COPY --from=builder /opt/tiktoken /opt/tiktoken
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken

# Install uv using the official method
COPY --from=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563 /uv /uvx /bin/

# Set the working directory in the container to /app
WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY --from=builder /app /app

# Copy built frontend from builder stage
COPY --from=builder /app/frontend/.next/standalone /app/frontend/
COPY --from=builder /app/frontend/.next/static /app/frontend/.next/static
COPY --from=builder /app/frontend/public /app/frontend/public

# Expose ports for Frontend and API
EXPOSE 8502 5055

RUN mkdir -p /app/data

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log directories
RUN mkdir -p /var/log/supervisor

# Runtime API URL Configuration
# The API_URL environment variable can be set at container runtime to configure
# where the frontend should connect to the API. This allows the same Docker image
# to work in different deployment scenarios without rebuilding.
#
# If not set, the system will auto-detect based on incoming requests.
# Set API_URL when using reverse proxies or custom domains.
#
# Example: docker run -e API_URL=https://your-domain.com/api ...

# De commit van de fork waaruit dit image is gebouwd (fork_commit(), F2-10);
# start.sh geeft hem mee. De HEAD van de host geldt alleen voor de flows.
# Achteraan, zodat een nieuwe commit de grote lagen niet ongeldig maakt.
ARG SIGNAAL_FORK_COMMIT=onbekend
LABEL org.opencontainers.image.revision=$SIGNAAL_FORK_COMMIT
ENV SIGNAAL_FORK_COMMIT=$SIGNAAL_FORK_COMMIT

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
