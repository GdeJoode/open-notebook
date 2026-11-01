"""Kleine client voor de API van Langflow 1.13.0 (F2-12, ADR 0017 A).

Alleen de standaardbibliotheek: het script draait in de container ``langflow``
en in de tests zonder extra pakketten. Aanmelden gaat met de superuser; het
wachtwoord komt uit ``LANGFLOW_SUPERUSER_PASSWORD`` (gezet door het startscript)
of uit het Docker secret ``/run/secrets/langflow_superuser``. Automatisch
aanmelden staat uit (ADR 0017 A, optie 2).

Gebruikte eindpunten (proeven.md P2, broncode van 1.13.0):

- ``POST /api/v1/login`` (formulier) geeft een toegangstoken;
- ``POST /api/v1/flows/upload/`` (multipart, één JSON-bestand) laadt een flow;
  een flow met een vaste ``id`` wordt bijgewerkt (upsert);
- ``POST /api/v1/flows/download/`` met één id geeft de flow als JSON, al
  genormaliseerd door Langflow (zonder ``updated_at``, ``user_id`` en dergelijke);
- ``GET /api/v1/triggers?flow_id=`` en ``POST /api/v1/triggers/{id}/enable``
  zetten de schema-trigger aan; de interface kan dat niet (P2, K2).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

STANDAARD_URL = "http://127.0.0.1:7860"
WACHTWOORD_BESTAND = Path("/run/secrets/langflow_superuser")


class LangflowFout(RuntimeError):
    """Langflow gaf een onverwachte status. Zonder antwoordtekst: die kan inhoud bevatten."""

    def __init__(self, wat: str, status: int) -> None:
        super().__init__(f"{wat}: HTTP {status}")
        self.status = status


def wachtwoord() -> str:
    waarde = os.environ.get("LANGFLOW_SUPERUSER_PASSWORD")
    if not waarde and WACHTWOORD_BESTAND.is_file():
        waarde = WACHTWOORD_BESTAND.read_text(encoding="utf-8").strip()
    if not waarde:
        raise LangflowFout("geen wachtwoord van de superuser", 0)
    return waarde


class LangflowApi:
    """Aanroepen van de API van één Langflow, aangemeld als superuser."""

    def __init__(self, url: Optional[str] = None, gebruiker: Optional[str] = None,
                 wachtwoord_: Optional[str] = None, timeout: float = 120.0) -> None:
        self.url = (url or os.environ.get("SIGNAAL_LANGFLOW_URL") or STANDAARD_URL).rstrip("/")
        self.gebruiker = gebruiker or os.environ.get("LANGFLOW_SUPERUSER", "")
        self._wachtwoord = wachtwoord_
        self.timeout = timeout
        self._token: Optional[str] = None

    # ------------------------------------------------------------------ HTTP
    def _verzoek(self, methode: str, pad: str, *, body: Any = None, data: Optional[bytes] = None,
                 headers: Optional[dict] = None, token: bool = True) -> tuple[int, Any]:
        kop = {"Accept": "application/json", **(headers or {})}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            kop["Content-Type"] = "application/json"
        if token:
            kop["Authorization"] = f"Bearer {self.token()}"
        verzoek = urllib.request.Request(self.url + pad, data=data, headers=kop, method=methode)
        # Geen proxy uit de omgeving: alleen het eigen adres.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(verzoek, timeout=self.timeout) as antwoord:
                tekst = antwoord.read()
                status = antwoord.status
        except urllib.error.HTTPError as fout:
            tekst, status = fout.read(), fout.code
        try:
            return status, (json.loads(tekst) if tekst else None)
        except ValueError:
            return status, None

    def gezond(self) -> bool:
        try:
            status, _ = self._verzoek("GET", "/health_check", token=False)
        except (OSError, urllib.error.URLError):
            return False
        return status == 200

    def wacht(self, seconden: float, pauze: float = 2.0) -> bool:
        einde = time.monotonic() + seconden
        while time.monotonic() < einde:
            if self.gezond():
                return True
            time.sleep(pauze)
        return False

    def token(self, pogingen: int = 6, pauze: float = 15.0) -> str:
        """Aanmelden; bij HTTP 429 (Langflow staat 5 aanmeldingen per minuut toe) opnieuw."""
        if self._token is None:
            formulier = urllib.parse.urlencode({
                "username": self.gebruiker,
                "password": self._wachtwoord or wachtwoord(),
            }).encode("utf-8")
            for poging in range(1, pogingen + 1):
                status, antwoord = self._verzoek(
                    "POST", "/api/v1/login", data=formulier, token=False,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
                if status != 429 or poging == pogingen:
                    break
                time.sleep(pauze)
            if status != 200 or not isinstance(antwoord, dict) or "access_token" not in antwoord:
                raise LangflowFout("aanmelden", status)
            self._token = antwoord["access_token"]
        return self._token

    # ------------------------------------------------------------------ flows
    def upload(self, flow: dict, bestandsnaam: str = "flow.json") -> list[dict]:
        grens = uuid.uuid4().hex
        inhoud = json.dumps(flow, ensure_ascii=False).encode("utf-8")
        data = (
            f"--{grens}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{bestandsnaam}\"\r\nContent-Type: application/json\r\n\r\n"
        ).encode("utf-8") + inhoud + f"\r\n--{grens}--\r\n".encode("utf-8")
        status, antwoord = self._verzoek(
            "POST", "/api/v1/flows/upload/", data=data,
            headers={"Content-Type": f"multipart/form-data; boundary={grens}"})
        if status not in (200, 201) or not isinstance(antwoord, list):
            raise LangflowFout(f"flow {bestandsnaam} laden", status)
        return antwoord

    def download(self, flow_id: str) -> dict:
        status, antwoord = self._verzoek("POST", "/api/v1/flows/download/", body=[flow_id])
        if status != 200 or not isinstance(antwoord, dict):
            raise LangflowFout(f"flow {flow_id} exporteren", status)
        return antwoord

    def bestaat(self, flow_id: str) -> bool:
        status, _ = self._verzoek("GET", f"/api/v1/flows/{flow_id}")
        if status not in (200, 404):
            raise LangflowFout(f"flow {flow_id} opvragen", status)
        return status == 200

    def verwijder(self, flow_id: str) -> None:
        status, _ = self._verzoek("DELETE", f"/api/v1/flows/{flow_id}")
        if status not in (200, 204, 404):
            raise LangflowFout(f"flow {flow_id} verwijderen", status)

    # ------------------------------------------------------------------ triggers
    def triggers(self, flow_id: str) -> list[dict]:
        status, antwoord = self._verzoek(
            "GET", f"/api/v1/triggers?flow_id={urllib.parse.quote(flow_id)}")
        if status != 200 or not isinstance(antwoord, list):
            raise LangflowFout(f"triggers van {flow_id}", status)
        return antwoord

    def zet_aan(self, trigger_id: str) -> dict:
        status, antwoord = self._verzoek("POST", f"/api/v1/triggers/{trigger_id}/enable")
        if status != 200 or not isinstance(antwoord, dict):
            raise LangflowFout(f"trigger {trigger_id} aanzetten", status)
        return antwoord
