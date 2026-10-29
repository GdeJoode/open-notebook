# P4: open-notebook-mcp 0.3.0 en de MCP-client mcp 1.30.0, plus de hulpscripts
# (docs/signaalsysteem/proeven.md, P4). Gebouwd met internet, draait op "verwerking".
FROM python:3.12.16-slim-bookworm
RUN pip install --no-cache-dir open-notebook-mcp==0.3.0 mcp==1.30.0 \
    && pip freeze > /proef-pip-freeze.txt
COPY p4/start_mcp.py p4/vul_testdatabase.py p4/mcp_lektest.py /proef/
