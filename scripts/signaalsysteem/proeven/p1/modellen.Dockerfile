# Downloadcontainer voor P1 en P3 (docs/signaalsysteem/proeven.md 1.7): modellen en testaudio.
FROM python:3.12.16-slim-bookworm
RUN pip install --no-cache-dir huggingface_hub==0.35.3 feedparser==6.0.14
COPY p1/modellen_downloaden.py /proef/modellen_downloaden.py
ENTRYPOINT ["python", "/proef/modellen_downloaden.py"]
