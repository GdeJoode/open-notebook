"""Maakt ``rondgang.json``: een export van Langflow 1.13.0 voor de tests van F2-12.

Draait in de container langflow (project signaal), met de sjablonen van de
kerncomponenten uit ``GET /api/v1/all``: een schema-trigger (één keer per jaar,
inhaalbeleid ``skip``) en TextInput naar TextOutput. Alleen typen van
``config/langflow-componenten.yaml``; geen gegevens, geen geheimen. De flow wordt
na de export weer verwijderd. Gebruik::

    docker exec -i <langflow> python - < services/langflow/tests/fixtures/maak_rondgang.py \\
        > services/langflow/tests/fixtures/rondgang.json
"""

import json
import sys

sys.path.insert(0, "/app/signaal")
from flows_exporteren import normaliseer  # noqa: E402
from langflow_api import LangflowApi  # noqa: E402

FLOW_ID = "f2120000-0000-4000-8000-000000000001"
api = LangflowApi()
_, alle = api._verzoek("GET", "/api/v1/all")
sjabloon = {naam: comp for groep in alle.values() if isinstance(groep, dict)
            for naam, comp in groep.items()}


def knoop(soort: str, nid: str, x: int, waarden: dict) -> dict:
    node = json.loads(json.dumps(sjabloon[soort]))
    for veld, waarde in waarden.items():
        node["template"][veld]["value"] = waarde
    return {"id": nid, "type": "genericNode", "position": {"x": x, "y": 0},
            "data": {"id": nid, "type": soort, "node": node}}


def handle(d: dict) -> str:
    return json.dumps(d, separators=(",", ":")).replace('"', "œ")


trigger = knoop("ScheduleTrigger", "ScheduleTrigger-f2120", 0,
                {"cron_expression": "0 3 1 1 *", "timezone": "Europe/Amsterdam",
                 "catchup_policy": "skip"})
invoer = knoop("TextInput", "TextInput-f2120", 300, {"input_value": "rondgang"})
uitvoer = knoop("TextOutput", "TextOutput-f2120", 600, {})
uit = next(o for o in invoer["data"]["node"]["outputs"] if o["name"] == "text")
veld = uitvoer["data"]["node"]["template"]["input_value"]
sh = {"dataType": "TextInput", "id": invoer["id"], "name": "text", "output_types": uit["types"]}
th = {"fieldName": "input_value", "id": uitvoer["id"], "inputTypes": veld.get("input_types") or [],
      "type": veld.get("type", "str")}
rand = {"id": f"xy-edge__{invoer['id']}-{uitvoer['id']}", "source": invoer["id"],
        "target": uitvoer["id"], "sourceHandle": handle(sh), "targetHandle": handle(th),
        "data": {"sourceHandle": sh, "targetHandle": th}}

flow = {"id": FLOW_ID, "name": "F2-12 rondgang (test)", "description": "Testflow F2-12",
        "data": {"nodes": [trigger, invoer, uitvoer], "edges": [rand],
                 "viewport": {"x": 0, "y": 0, "zoom": 1}}}
api.verwijder(FLOW_ID)
api.upload(flow, "rondgang.json")
export = normaliseer(api.download(FLOW_ID))
api.verwijder(FLOW_ID)
print(json.dumps(export, indent=2, sort_keys=True, ensure_ascii=False))
