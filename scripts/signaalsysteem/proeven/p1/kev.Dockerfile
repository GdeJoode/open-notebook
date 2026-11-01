# Kev (jaredpalmer/kev, tag kev-1.0), lokaal gebouwd (docs/signaalsysteem/proeven.md, P1).
# Kev heeft geen eigen image en staat niet op PyPI ("pip install kev" is een ander pakket).
# PyTorch komt via het lockbestand van Kev; vastgelegd in /kev-pip-freeze.txt
# (CUDA 12.8 of hoger nodig voor Blackwell).
FROM ghcr.io/astral-sh/uv:0.11.13 AS uv

FROM python:3.13.18-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch kev-1.0 https://github.com/jaredpalmer/kev.git /opt/kev
WORKDIR /opt/kev
ENV UV_PROJECT_ENVIRONMENT=/opt/kev/.venv UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/local/bin/python3.13
RUN uv sync --extra serve && uv pip freeze > /kev-pip-freeze.txt
ENV PATH=/opt/kev/.venv/bin:$PATH
