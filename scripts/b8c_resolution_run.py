"""B.8c overnight orchestration: sequential qwen extraction over the 4 Convenant
sources, then a cross-document resolution snapshot. Runs detached inside the
open_notebook container; writes progress to /tmp/b8c_overnight.log (flushed)."""
import asyncio
import json
import time
import urllib.request
import urllib.parse

from surrealdb_service.connection import execute_query
from surrealdb_service.config import get_config

conf = get_config()
API = "http://localhost:5055/api"
LOG = "/tmp/b8c_overnight.log"

# (source_id, short label)
SOURCES = [
    ("source:0o96pew25fovpoom5n6r", "Het Hogeland"),
    ("source:hjrb1cjvzv86oyaojblj", "Noord-Holland Noord"),
    ("source:dndibxmjveoxk7tfqfsl", "Midden-Limburg"),
    ("source:052dtl7jrwu1czlpnui4", "Zuidwest-Friesland"),
]
BC6XA_JOB = "job:26keyr7ls68gmx2qyqwn"  # the in-flight bc6xa run to wait for


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


async def db(q, p=None):
    return await execute_query(q, p or {}, conf)


async def ecount(sid):
    r = await db(
        "SELECT count() FROM entity WHERE $s IN source_documents GROUP ALL",
        {"s": sid},
    )
    return r[0]["count"] if r else 0


async def job_status(jid):
    r = await db("SELECT status FROM $j", {"j": jid}) if False else \
        await db(f"SELECT status FROM {jid}")
    return r[0]["status"] if r else "unknown"


async def source_parsed(sid):
    r = await db(f"SELECT full_text != NONE AS parsed, title FROM {sid}")
    return bool(r and r[0].get("parsed"))


def trigger_extract(sid):
    enc = urllib.parse.quote(sid, safe="")
    req = urllib.request.Request(
        f"{API}/sources/{enc}/run-entities",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("command_id")


async def wait_job(jid, label, max_s=9000):
    """Poll a job until terminal. Returns final status."""
    for _ in range(max_s // 15):
        st = await job_status(jid)
        if st in ("completed", "failed", "dead_letter", "cancelled"):
            return st
        await asyncio.sleep(15)
    return "timeout"


async def main():
    log("=== B.8c overnight run start ===")
    # 1. wait for the in-flight bc6xa extraction to finish (no parallel qwen)
    log(f"waiting for bc6xa job {BC6XA_JOB} ...")
    st = await wait_job(BC6XA_JOB, "bc6xa")
    bc = await ecount("source:bc6xax5piw8cntg5nomc")
    log(f"bc6xa job -> {st}; bc6xa persisted entities = {bc}")

    # 2. per Convenant source: wait parsed -> extract -> wait done -> count
    results = []
    for sid, label in SOURCES:
        log(f"--- {label} ({sid}) ---")
        # wait for parse (docling) up to 20 min
        parsed = False
        for _ in range(80):
            if await source_parsed(sid):
                parsed = True
                break
            await asyncio.sleep(15)
        if not parsed:
            log(f"  {label}: NOT parsed after 20m — skipping")
            results.append((label, sid, "parse_timeout", 0))
            continue
        log(f"  {label}: parsed; triggering extraction")
        try:
            jid = trigger_extract(sid)
        except Exception as e:
            log(f"  {label}: trigger failed: {e}")
            results.append((label, sid, "trigger_failed", 0))
            continue
        st = await wait_job(jid, label)
        n = await ecount(sid)
        log(f"  {label}: extraction {st}; persisted entities = {n}")
        results.append((label, sid, st, n))

    # 3. resolution snapshot across the Convenant set
    log("=== resolution snapshot ===")
    conv_ids = [sid for sid, _ in SOURCES]
    # entities appearing in >1 Convenant source (shared canonical entities)
    shared = await db(
        "SELECT canonical_name, entity_type, count() AS docs FROM entity "
        "WHERE source_documents CONTAINSANY $ids "
        "GROUP BY canonical_name, entity_type",
        {"ids": conv_ids},
    )
    # crude cross-doc: count canonical names that span >=2 of our sources
    multi = 0
    examples = []
    for sid, _ in SOURCES:
        pass
    # recompute per-entity doc-span over our 4 sources
    rows = await db(
        "SELECT canonical_name, entity_type, source_documents FROM entity "
        "WHERE source_documents CONTAINSANY $ids",
        {"ids": conv_ids},
    )
    span = {}
    for r in (rows or []):
        key = (r.get("canonical_name"), r.get("entity_type"))
        docs = set(str(d) for d in (r.get("source_documents") or []) if str(d) in conv_ids)
        span.setdefault(key, set()).update(docs)
    multi = sum(1 for k, v in span.items() if len(v) >= 2)
    top = sorted(span.items(), key=lambda kv: -len(kv[1]))[:10]
    log(f"distinct canonical entities across the 4 docs: {len(span)}")
    log(f"entities spanning >=2 docs (resolved cross-doc): {multi}")
    for (cn, et), docs in top:
        log(f"  shared[{len(docs)} docs] {et}: {cn}")
    log("=== B.8c overnight run COMPLETE ===")
    with open("/tmp/b8c_results.json", "w") as f:
        json.dump(
            {"per_doc": results, "distinct": len(span), "multi_doc": multi},
            f,
            default=str,
        )


asyncio.run(main())
