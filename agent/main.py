"""NYC Vision Agent — direction-aware camera intelligence for Manhattan.

Deployed on Google Cloud Run. Deliberately light: no torch, no model weights,
so the image stays small and cold starts are quick.

Detection runs at the edge (see ../collector), which POSTs each sweep to
/api/ingest. This service owns everything else:

  * the live map + time-series UI
  * the 311 verification agent, which finds cameras that can *actually see* a
    complaint using our camera-heading estimates, then asks Gemini whether the
    reported condition is still visible

The 311 agent runs entirely here, so it keeps working even if the edge
collector goes away mid-demo.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"


def _load_dotenv() -> None:
    """Local convenience only. On Cloud Run there is no .env and credentials
    come from the service account, so this is a no-op there."""
    for candidate in (HERE / ".env", HERE.parent / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
        break


_load_dotenv()

COUNT_KEYS = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

MAX_CYCLES = int(os.environ.get("MAX_CYCLES", "400"))
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")
VISION_HALF_ANGLE = float(os.environ.get("VISION_HALF_ANGLE", "55"))
FRAME_TIMEOUT = 15

HEADINGS = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}


def _load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8-sig"))


CAMERAS: List[Dict] = _load("manhattan_cams.json")
DIRECTIONS: Dict[str, Dict] = _load("combined_directions.json")

for _c in CAMERAS:
    _d = DIRECTIONS.get(_c["id"])
    _c["dir"] = _d["dir"] if _d else None
    _c["dir_source"] = _d["source"] if _d else None

CAM_BY_ID = {c["id"]: c for c in CAMERAS}

# --- geo ------------------------------------------------------------------


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def off_axis(cam: Dict, lat: float, lon: float) -> Optional[float]:
    """Degrees between where the camera looks and where the incident is.

    None when we have no heading for that camera. This is the difference
    between 'nearest camera' and 'camera that can see it'.
    """
    if not cam.get("dir"):
        return None
    return angle_gap(
        bearing_deg(cam["latitude"], cam["longitude"], lat, lon),
        HEADINGS[cam["dir"]],
    )


def cameras_that_can_see(lat: float, lon: float, radius_m: float = 350,
                         top: int = 4) -> List[Dict]:
    out = []
    for cam in CAMERAS:
        d = haversine_m(lat, lon, cam["latitude"], cam["longitude"])
        if d > radius_m:
            continue
        off = off_axis(cam, lat, lon)
        out.append({
            "id": cam["id"], "name": cam["name"], "url": cam["url"],
            "latitude": cam["latitude"], "longitude": cam["longitude"],
            "distance_m": round(d), "dir": cam.get("dir"),
            "dir_source": cam.get("dir_source"),
            "off_axis_deg": None if off is None else round(off),
            "in_view": off is not None and off <= VISION_HALF_ANGLE,
        })
    out.sort(key=lambda c: (not c["in_view"], c["distance_m"]))
    return out[:top]


# --- ingested sweeps ------------------------------------------------------

CYCLES: deque = deque(maxlen=MAX_CYCLES)
STATE: Dict[str, Any] = {
    "started": time.time(), "cycles_ingested": 0, "last_cycle": None,
    "edge": None, "last_error": None,
}
_lock = threading.Lock()


def snapshot_to_gcs(cycle: Dict) -> None:
    if not GCS_BUCKET:
        return
    try:
        from google.cloud import storage
        blob = storage.Client().bucket(GCS_BUCKET).blob(f"cycles/{int(cycle['ts'])}.json")
        blob.upload_from_string(json.dumps(cycle), content_type="application/json")
    except Exception as e:
        STATE["last_error"] = f"gcs: {e}"


# --- gemini ---------------------------------------------------------------


def gemini_verify(frame: bytes, complaint_type: str, descriptor: str,
                  address: str) -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    else:
        key = os.environ.get("GEMINI_API_KEY_HACK") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise HTTPException(500, "no Gemini credentials configured")
        client = genai.Client(api_key=key)

    prompt = (
        "You are verifying a NYC 311 complaint against a live traffic-camera frame.\n"
        f"Complaint type: {complaint_type}\n"
        f"Descriptor: {descriptor}\n"
        f"Reported near: {address}\n\n"
        "The image is a low-resolution municipal traffic camera still (352x240). "
        "Judge only what is genuinely visible; do not speculate. If the frame "
        "cannot settle the question, say so plainly.\n\n"
        "Reply as strict JSON with keys: verdict (one of 'visible', "
        "'not_visible', 'inconclusive'), confidence (0-1), reasoning (<=30 "
        "words), scene (<=20 words describing the frame)."
    )

    resp = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        contents=[types.Part.from_bytes(data=frame, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json"),
    )
    try:
        return json.loads(resp.text)
    except Exception:
        return {"verdict": "inconclusive", "confidence": 0.0,
                "reasoning": "model returned unparseable output",
                "scene": (resp.text or "")[:120]}


SOCRATA_311 = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"


def recent_311(limit: int = 25, complaint: Optional[str] = None) -> List[Dict]:
    where = ["borough='MANHATTAN'", "latitude IS NOT NULL"]
    if complaint:
        where.append("complaint_type='%s'" % complaint.replace("'", "''"))
    r = requests.get(SOCRATA_311, timeout=30, params={
        "$where": " AND ".join(where),
        "$order": "created_date DESC",
        "$limit": str(limit),
        "$select": "unique_key,created_date,complaint_type,descriptor,"
                   "incident_address,status,latitude,longitude",
    })
    r.raise_for_status()
    return r.json()


# --- api ------------------------------------------------------------------

app = FastAPI(title="NYC Vision Agent")


class Sweep(BaseModel):
    ts: float
    ok: int
    attempted: int
    fetch_s: float = 0
    detect_s: float = 0
    device: str = "unknown"
    totals: Dict[str, int]
    counts: Dict[str, Dict[str, int]]


@app.post("/api/ingest")
def api_ingest(sweep: Sweep, x_ingest_token: str = Header(default="")):
    if INGEST_TOKEN and x_ingest_token != INGEST_TOKEN:
        raise HTTPException(401, "bad ingest token")
    cycle = sweep.model_dump()
    with _lock:
        CYCLES.append(cycle)
        STATE["cycles_ingested"] += 1
        STATE["last_cycle"] = cycle["ts"]
        STATE["edge"] = cycle.get("device")
    snapshot_to_gcs(cycle)
    return {"ok": True, "buffered": len(CYCLES)}


@app.get("/api/health")
@app.get("/healthz")
def healthz():
    age = None
    if STATE["last_cycle"]:
        age = round(time.time() - STATE["last_cycle"])
    return {"ok": True, **STATE, "cycles_buffered": len(CYCLES),
            "seconds_since_last_sweep": age, "cameras": len(CAMERAS),
            "with_heading": sum(1 for c in CAMERAS if c["dir"])}


@app.get("/api/cameras")
def api_cameras():
    return [{k: c[k] for k in
             ("id", "name", "latitude", "longitude", "url", "dir", "dir_source")}
            for c in CAMERAS]


@app.get("/api/latest")
def api_latest():
    with _lock:
        if not CYCLES:
            return JSONResponse({"ready": False, "state": STATE}, status_code=202)
        return {"ready": True, **CYCLES[-1]}


@app.get("/api/cycles")
def api_cycles(limit: int = 400):
    """Every buffered sweep, counts included, so the UI can scrub history
    without refetching each one."""
    with _lock:
        return list(CYCLES)[-limit:]


@app.get("/api/series")
def api_series(camera_id: Optional[str] = None):
    with _lock:
        cycles = list(CYCLES)
    if camera_id:
        blank = {k: 0 for k in COUNT_KEYS}
        return [{"ts": c["ts"], **c["counts"].get(camera_id, blank)} for c in cycles]
    return [{"ts": c["ts"], **c["totals"]} for c in cycles]


@app.get("/api/311")
def api_311(limit: int = 25, complaint_type: Optional[str] = None):
    try:
        rows = recent_311(limit=limit, complaint=complaint_type)
    except Exception as e:
        raise HTTPException(502, f"311 fetch failed: {e}")
    for r in rows:
        try:
            r["cameras"] = cameras_that_can_see(float(r["latitude"]), float(r["longitude"]))
        except Exception:
            r["cameras"] = []
    return rows


class VerifyRequest(BaseModel):
    latitude: float
    longitude: float
    complaint_type: str = "Street Condition"
    descriptor: str = ""
    address: str = ""


@app.post("/api/verify")
def api_verify(req: VerifyRequest):
    cams = cameras_that_can_see(req.latitude, req.longitude)
    if not cams:
        return {"verified": False, "reason": "no camera within range", "cameras": []}

    best = cams[0]
    try:
        r = requests.get(best["url"], timeout=FRAME_TIMEOUT)
        frame = r.content if r.status_code == 200 else None
    except Exception:
        frame = None
    if not frame:
        return {"verified": False, "reason": "camera frame unavailable",
                "camera": best, "cameras": cams}

    # Surface the real reason rather than a bare 500 — on Cloud Run the usual
    # causes are aiplatform.googleapis.com not enabled, or the service account
    # missing roles/aiplatform.user, and both are one command to fix.
    try:
        assessment = gemini_verify(frame, req.complaint_type,
                                   req.descriptor, req.address)
    except Exception as e:
        STATE["last_error"] = f"gemini: {type(e).__name__}: {e}"
        return {"verified": False, "camera": best, "cameras": cams,
                "reason": "model call failed",
                "error": f"{type(e).__name__}: {e}"[:600],
                "auth_mode": "vertex" if os.environ.get(
                    "GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
                    else "api-key",
                "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                "checked_at": time.time()}

    return {"verified": True, "camera": best, "cameras": cams,
            "assessment": assessment, "checked_at": time.time()}


app.mount("/", StaticFiles(directory=str(HERE / "static"), html=True), name="static")
