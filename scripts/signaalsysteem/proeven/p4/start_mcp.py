"""Start open-notebook-mcp 0.3.0 met streamable HTTP onder mcp==1.30.0.

Het officiele startpunt (open_notebook_mcp.main) roept mcp.run(transport=...,
host=..., port=..., path=..., ...) aan. FastMCP.run in mcp 1.30.0 kent alleen
transport en mount_path, dus dat startpunt faalt met een TypeError. Het script
p4_mcp_classificatiefilter.sh legt dat vast als bevinding (aanname A2). Dit
startbestand zet dezelfde waarden via mcp.settings en verandert verder niets
aan de server of de tools.

Het wachtwoord komt uit het Compose-secret en gaat alleen in de omgeving van
dit proces.
"""

import os

with open("/run/secrets/notebook_wachtwoord") as f:
    os.environ["OPEN_NOTEBOOK_PASSWORD"] = f.read().strip()

from open_notebook_mcp.server import mcp  # noqa: E402

mcp.settings.host = os.getenv("HOST", "127.0.0.1")
mcp.settings.port = int(os.getenv("PORT", "8000"))
mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp")
mcp.settings.stateless_http = os.getenv("STATELESS_HTTP", "1") == "1"
mcp.settings.json_response = os.getenv("JSON_RESPONSE", "1") == "1"
# FastMCP zet bij host 127.0.0.1 (de standaard bij het aanmaken) een bescherming
# tegen DNS-rebinding met alleen localhost als toegestane Host. Het origineel
# geeft host mee aan run(); hier hetzelfde effect: geen Host-beperking.
mcp.settings.transport_security = None
mcp.run(transport="streamable-http")
