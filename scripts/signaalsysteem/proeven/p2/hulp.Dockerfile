# P2: hulpcontainer met de Langflow-aanroepen en de terugval met APScheduler
# (docs/signaalsysteem/proeven.md, P2). Gebouwd met internet, draait op "verwerking".
FROM python:3.12.16-slim-bookworm
RUN pip install --no-cache-dir APScheduler==3.11.3 SQLAlchemy==2.0.43 \
    && pip freeze > /proef-pip-freeze.txt
# Gedeeld volume met Langflow (uid 1000): schrijfbaar voor iedereen, ook als dit
# image het volume als eerste aanmaakt.
RUN mkdir -p /proef/ticks /proef/planner && chmod 0777 /proef/ticks /proef/planner
COPY p2/langflow_api.py p2/apscheduler_terugval.py /proef/
