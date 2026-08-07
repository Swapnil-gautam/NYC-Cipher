"""Replay already-collected sweeps into a running agent.

Useful twice: to populate a freshly deployed Cloud Run service with everything
gathered before it existed, and to re-seed after a restart.

  INGEST_URL=http://127.0.0.1:8099/api/ingest python backfill.py
"""
import json
import os
import pathlib
import sys

import requests

SWEEPS = pathlib.Path(__file__).parent / "data" / "sweeps.jsonl"
URL = os.environ.get("INGEST_URL", "http://127.0.0.1:8099/api/ingest")
TOKEN = os.environ.get("INGEST_TOKEN", "")

if not SWEEPS.exists():
    sys.exit(f"no sweeps at {SWEEPS}")

rows = [json.loads(l) for l in SWEEPS.read_text(encoding="utf-8").splitlines() if l.strip()]
rows.sort(key=lambda r: r["ts"])
print(f"replaying {len(rows)} sweeps -> {URL}")

ok = 0
for r in rows:
    try:
        resp = requests.post(URL, json=r, timeout=30,
                             headers={"X-Ingest-Token": TOKEN})
        if resp.status_code == 200:
            ok += 1
        else:
            print(f"  {int(r['ts'])}: HTTP {resp.status_code} {resp.text[:120]}")
    except Exception as e:
        print(f"  {int(r['ts'])}: {type(e).__name__}")
print(f"done: {ok}/{len(rows)} accepted")
