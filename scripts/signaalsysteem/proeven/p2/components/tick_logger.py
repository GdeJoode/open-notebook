"""TickLogger: Langflow-component voor P2 (docs/signaalsysteem/proeven.md, P2).

Schrijft per uitvoering één JSON-regel naar PROEF_TICKS (proefvolume): het label,
de werkelijke tijd en het triggerevent (met daarin scheduled_at en missed_ticks
als de schema-trigger de flow start). Zonder event (een gewone API-run, zoals de
terugval met APScheduler) is het event leeg.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, MessageTextInput, Output
from lfx.schema.data import Data


class TickLogger(Component):
    display_name = "TickLogger"
    description = "Schrijft per uitvoering een regel naar ticks.jsonl (proef P2)."
    name = "TickLogger"
    icon = "Clock"

    inputs = [
        DataInput(name="event", display_name="Event", required=False),
        MessageTextInput(name="label", display_name="Label", value="proef"),
    ]
    outputs = [Output(display_name="Resultaat", name="resultaat", method="log_tick")]

    def log_tick(self) -> Data:
        event = self.event
        if isinstance(event, list):
            event = event[0] if event else None
        gegevens = event.data if isinstance(event, Data) else event
        regel = {
            "label": self.label,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "event": gegevens,
        }
        with open(os.environ.get("PROEF_TICKS", "/proef/ticks/ticks.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(regel, default=str) + "\n")
        return Data(data=regel)
