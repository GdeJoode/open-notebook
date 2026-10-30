"""P2: aanroepen van de Langflow-API (docs/signaalsysteem/proeven.md, P2).

Draait in de hulpcontainer op het netwerk "verwerking". Elke opdracht schrijft
JSON naar stdout. Het wachtwoord komt uit het Compose-secret.

  inventaris                  2b: schema-component in /api/v1/all, /api/v1/config
  apisleutel                  2a: inloggen en een API-sleutel maken (sleutel op stdout)
  opzet                       2c en 2f: flows maken, triggers ophalen en aanzetten
  beleid <flow_id> <beleid>   catchup_policy wijzigen (flow opslaan) en trigger weer aanzetten
  triggers <flow_id>          toestand van de triggers van een flow
  analyse <bestand> <label> <van_iso> <tot_iso> [herstart_iso]
                              ticks tellen, vertraging en inhaalruns bepalen
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

URL = os.environ.get("LANGFLOW_URL", "http://langflow:7860")
GEBRUIKER = os.environ.get("LANGFLOW_SUPERUSER", "proefbeheer")


def wachtwoord() -> str:
    with open("/run/secrets/langflow_wachtwoord") as f:
        return f.read().strip()


def http(method: str, pad: str, token: str | None = None, body=None, form: dict | None = None):
    headers = {"Accept": "application/json"}
    data = None
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(URL + pad, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            tekst = r.read()
            return r.status, (json.loads(tekst) if tekst else None)
    except urllib.error.HTTPError as e:
        tekst = e.read()
        try:
            return e.code, json.loads(tekst)
        except ValueError:
            return e.code, tekst.decode("utf-8", "replace")[:500]


def login() -> str:
    status, res = http("POST", "/api/v1/login", form={"username": GEBRUIKER, "password": wachtwoord()})
    if status != 200:
        raise SystemExit(f"login mislukt: HTTP {status}")
    return res["access_token"]


def alle_componenten(token: str) -> dict:
    status, res = http("GET", "/api/v1/all", token)
    if status != 200:
        raise SystemExit(f"/api/v1/all: HTTP {status}")
    return res


def zoek_component(alle: dict, naam: str) -> tuple[str | None, str | None, dict | None]:
    """Zoek een component op naam. Componenten uit LANGFLOW_COMPONENTS_PATH staan in
    1.13.0 onder een sleutel als "ext:proef:TickLogger@extra"; die sleutel is ook het
    knooptype in de flow."""
    for categorie, comps in alle.items():
        if not isinstance(comps, dict):
            continue
        for sleutel, sjabloon in comps.items():
            if sleutel == naam or (sleutel.startswith("ext:") and sleutel.split(":")[-1].split("@")[0] == naam):
                return categorie, sleutel, sjabloon
    return None, None, None


def inventaris() -> dict:
    token = login()
    alle = alle_componenten(token)
    cat_s, _, schema = zoek_component(alle, "ScheduleTrigger")
    cat_t, type_t, tick = zoek_component(alle, "TickLogger")
    status, config = http("GET", "/api/v1/config", token)
    sleutels = sorted(config.keys()) if isinstance(config, dict) else []
    return {
        "schedule_trigger_in_all": schema is not None,
        "schedule_trigger_categorie": cat_s,
        "categorieen": sorted(alle.keys()),
        "ticklogger_in_all": tick is not None,
        "ticklogger_categorie": cat_t,
        "ticklogger_type": type_t,
        "schedule_trigger_uitvoertypen": [o.get("types") for o in (schema or {}).get("outputs", [])],
        "config_status": status,
        "config_feature_flags": config.get("feature_flags") if isinstance(config, dict) else None,
        "config_sleutels_met_trigger": [k for k in sleutels if "trigger" in k.lower()],
    }


def apisleutel() -> str:
    token = login()
    status, res = http("POST", "/api/v1/api_key/", token, {"name": f"proef-{uuid.uuid4().hex[:6]}"})
    if status not in (200, 201):
        raise SystemExit(f"api_key: HTTP {status}")
    return res["api_key"]


def handle(d: dict) -> str:
    return json.dumps(d, separators=(",", ":")).replace('"', "œ")


def knoop(comp: str, sjabloon: dict, waarden: dict) -> dict:
    node = json.loads(json.dumps(sjabloon))
    for veld, waarde in waarden.items():
        node["template"][veld]["value"] = waarde
    kort = comp.split(":")[-1].split("@")[0]
    nid = f"{kort}-{uuid.uuid4().hex[:5]}"
    return {"id": nid, "type": "genericNode", "position": {"x": 0, "y": 0},
            "data": {"id": nid, "type": comp, "node": node}}


def rand(bron: dict, doel: dict) -> dict:
    uit = next(o for o in bron["data"]["node"]["outputs"] if o["name"] == "trigger_event")
    veld = doel["data"]["node"]["template"]["event"]
    sh = {"dataType": bron["data"]["type"], "id": bron["id"], "name": "trigger_event", "output_types": uit["types"]}
    th = {"fieldName": "event", "id": doel["id"], "inputTypes": veld.get("input_types") or [], "type": veld.get("type", "other")}
    return {
        "id": f"xy-edge__{bron['id']}-{doel['id']}",
        "source": bron["id"], "target": doel["id"],
        "sourceHandle": handle(sh), "targetHandle": handle(th),
        "data": {"sourceHandle": sh, "targetHandle": th},
    }


def maak_flow(token: str, naam: str, knopen: list[dict], randen: list[dict]) -> str:
    status, res = http("POST", "/api/v1/flows/", token,
                       {"name": naam, "description": "proef P2", "data": {"nodes": knopen, "edges": randen}})
    if status not in (200, 201):
        raise SystemExit(f"flow {naam}: HTTP {status}: {res}")
    return res["id"]


def triggers(token: str, flow_id: str) -> tuple[int, list]:
    status, res = http("GET", f"/api/v1/triggers?flow_id={flow_id}", token)
    return status, res if isinstance(res, list) else []


def kern(t: dict) -> dict:
    return {k: t.get(k) for k in ("id", "node_id", "kind", "state", "config", "next_fire_at", "last_error",
                                  "last_fired_at")}


def opzet() -> dict:
    token = login()
    alle = alle_componenten(token)
    _, type_s, schema = zoek_component(alle, "ScheduleTrigger")
    _, type_t, tick = zoek_component(alle, "TickLogger")
    if schema is None or tick is None:
        raise SystemExit("ScheduleTrigger of TickLogger ontbreekt in /api/v1/all")
    uit: dict = {}

    s1 = knoop(type_s, schema, {"cron_expression": "* * * * *", "timezone": "Europe/Amsterdam",
                                           "catchup_policy": "coalesce"})
    t1 = knoop(type_t, tick, {"label": "schema"})
    uit["flow_schema"] = maak_flow(token, "proef-schema-elke-minuut", [s1, t1], [rand(s1, t1)])

    s2 = knoop(type_s, schema, {"cron_expression": "0 8 * * 1-5", "timezone": "Europe/Amsterdam",
                                           "catchup_policy": "coalesce"})
    t2 = knoop(type_t, tick, {"label": "tijdzone"})
    uit["flow_tijdzone"] = maak_flow(token, "proef-schema-tijdzone", [s2, t2], [rand(s2, t2)])

    t3 = knoop(type_t, tick, {"label": "planner"})
    uit["flow_planner"] = maak_flow(token, "proef-planner-zonder-trigger", [t3], [])

    for sleutel in ("flow_schema", "flow_tijdzone"):
        st, lijst = triggers(token, uit[sleutel])
        uit[f"{sleutel}_triggers_status"] = st
        uit[f"{sleutel}_triggers_na_opslaan"] = [kern(t) for t in lijst]
        aangezet = []
        for t in lijst:
            s_en, res = http("POST", f"/api/v1/triggers/{t['id']}/enable", token)
            aangezet.append({"enable_status": s_en, **(kern(res) if isinstance(res, dict) else {"antwoord": res})})
        uit[f"{sleutel}_enable"] = aangezet
    return uit


def beleid(flow_id: str, nieuw: str) -> dict:
    token = login()
    status, flow = http("GET", f"/api/v1/flows/{flow_id}", token)
    if status != 200:
        raise SystemExit(f"flow ophalen: HTTP {status}")
    for n in flow["data"]["nodes"]:
        if n["data"]["type"].split(":")[-1].split("@")[0] == "ScheduleTrigger":
            n["data"]["node"]["template"]["catchup_policy"]["value"] = nieuw
    s_p, _ = http("PATCH", f"/api/v1/flows/{flow_id}", token, {"data": flow["data"]})
    _, lijst = triggers(token, flow_id)
    uit = {"patch_status": s_p, "na_opslaan": [kern(t) for t in lijst], "enable": []}
    for t in lijst:
        s_en, res = http("POST", f"/api/v1/triggers/{t['id']}/enable", token)
        uit["enable"].append({"enable_status": s_en, **(kern(res) if isinstance(res, dict) else {"antwoord": res})})
    return uit


def triggers_cmd(flow_id: str) -> dict:
    token = login()
    st, lijst = triggers(token, flow_id)
    return {"status": st, "triggers": [kern(t) for t in lijst]}


def iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def analyse(bestand: str, label: str, van: str, tot: str, herstart: str | None = None) -> dict:
    van_t, tot_t = iso(van), iso(tot)
    herstart_t = iso(herstart) if herstart else None
    regels = []
    if os.path.exists(bestand):
        with open(bestand, encoding="utf-8") as f:
            regels = [json.loads(r) for r in f if r.strip()]
    runs = []
    for r in regels:
        if "label" in r and r["label"] != label:
            continue
        werkelijk = iso(r["logged_at"])
        if not (van_t <= werkelijk <= tot_t):
            continue
        if "scheduled" in r:  # regel van de terugval
            # APScheduler meldt bij coalesce het laatst gemiste moment als gepland
            # moment; een run waarvan dat moment vóór de herstart ligt, is de inhaalrun.
            gepland, status = iso(r["scheduled"]), r.get("status")
            gemist = ["voor herstart"] if herstart_t and gepland < herstart_t <= werkelijk else []
        else:
            payload = ((r.get("event") or {}).get("payload") or {})
            gepland = iso(payload["scheduled_at"]) if payload.get("scheduled_at") else None
            gemist, status = payload.get("missed_ticks") or [], None
        vertraging = (werkelijk - gepland).total_seconds() if gepland else None
        runs.append({"gepland": gepland.isoformat() if gepland else None, "werkelijk": werkelijk.isoformat(),
                     "vertraging_s": vertraging, "gemist": len(gemist), "status": status})
    gepland_lijst = [r["gepland"] for r in runs if r["gepland"]]
    na = [r for r in runs if herstart_t and iso(r["werkelijk"]) >= herstart_t]
    inhaal = [r for r in na if r["gemist"] > 0 or (r["vertraging_s"] is not None and r["vertraging_s"] > 30)]
    return {
        "aantal": len(runs),
        "uniek_gepland": len(set(gepland_lijst)) == len(gepland_lijst),
        "zonder_gepland": sum(1 for r in runs if not r["gepland"]),
        "max_vertraging_s": max((r["vertraging_s"] for r in runs if r["vertraging_s"] is not None), default=None),
        "status_200": sum(1 for r in runs if r["status"] == 200),
        "inhaalruns": len(inhaal),
        "inhaal_gemist": [r["gemist"] for r in inhaal],
        "runs": runs,
    }


def main() -> None:
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "apisleutel":
        print(apisleutel())
        return
    res = {
        "inventaris": lambda: inventaris(),
        "opzet": lambda: opzet(),
        "beleid": lambda: beleid(*args),
        "triggers": lambda: triggers_cmd(*args),
        "analyse": lambda: analyse(*args),
    }[cmd]()
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
