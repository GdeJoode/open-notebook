# Build stage
FROM python:3.12-slim-bookworm AS builder

# Install uv using the official method
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies required for building certain Python packages
# Add Node.js 20.x LTS for building frontend
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    gcc g++ git make \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

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

# Return to app root
WORKDIR /app

# Runtime stage
FROM python:3.12-slim-bookworm AS runtime

# Install only runtime system dependencies (no build tools)
# Add Node.js 20.x LTS for running frontend
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ffmpeg \
    supervisor \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official method
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
