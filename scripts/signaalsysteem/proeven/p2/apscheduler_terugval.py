"""P2, stap 2g: terugval met APScheduler 3.11.3 (docs/signaalsysteem/proeven.md, P2).

CronTrigger elke minuut, coalesce=True, misfire_grace_time=300, SQLite-jobstore
in een proefvolume. Start de flow zonder triggerknooppunt via
POST /api/v1/run/{flow_id} met x-api-key. Schrijft per run een JSON-regel naar
PLANNER_LOG: gepland moment, werkelijke tijd en HTTP-status.

Bij een herstart blijft de job in de jobstore staan; de scheduler start gepauzeerd,
voegt de job alleen toe als hij ontbreekt en hervat dan. Zo worden gemiste
momenten volgens coalesce in één inhaalrun samengevoegd.
"""

from __future__ import annotations

import json
import os
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

LANGFLOW_URL = os.environ.get("LANGFLOW_URL", "http://langflow:7860")
FLOW_ID = os.environ["PLANNER_FLOW_ID"]
LOG = os.environ.get("PLANNER_LOG", "/proef/ticks/planner.jsonl")
JOBSTORE = os.environ.get("PLANNER_JOBSTORE", "sqlite:////proef/planner/jobs.sqlite")
with open("/run/secrets/langflow_api_sleutel") as f:
    API_KEY = f.read().strip()


def run_flow() -> int:
    req = urllib.request.Request(
        f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}?stream=false",
        data=json.dumps({"input_value": "", "input_type": "text", "output_type": "any"}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": API_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def luisteraar(event) -> None:
    regel = {
        "scheduled": event.scheduled_run_time.astimezone(timezone.utc).isoformat(),
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "status": getattr(event, "retval", None),
        "fout": repr(event.exception) if getattr(event, "exception", None) else None,
    }
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(regel) + "\n")


def main() -> None:
    sched = BackgroundScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=JOBSTORE)},
        job_defaults={"coalesce": True, "misfire_grace_time": 300, "max_instances": 1},
        timezone="Europe/Amsterdam",
    )
    sched.add_listener(luisteraar, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    sched.start(paused=True)
    if sched.get_job("tick") is None:
        sched.add_job(run_flow, CronTrigger.from_crontab("* * * * *", timezone="Europe/Amsterdam"), id="tick")
    sched.resume()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    while not stop:
        time.sleep(1)
    sched.shutdown(wait=False)


if __name__ == "__main__":
    main()
