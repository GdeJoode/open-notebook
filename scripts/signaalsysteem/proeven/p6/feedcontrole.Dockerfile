# P6: feedcontrole in een Python-container (docs/signaalsysteem/proeven.md, P6).
FROM python:3.12.16-slim-bookworm
RUN pip install --no-cache-dir feedparser==6.0.14
COPY p6/feedcontrole.py /proef/feedcontrole.py
ENTRYPOINT ["python", "/proef/feedcontrole.py"]
