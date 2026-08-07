"""Edge collector — runs on the laptop, not on Cloud Run.

Sweeps every Manhattan DOT camera on a cycle, saves the raw frames, counts
people / bikes / cars on the local GPU, and (optionally) POSTs each sweep to
the Cloud Run service for the live map.

Detection is here rather than in the container because the RTX 5070 does the
whole city in ~2s where Cloud Run's CPU needs ~12s, and because it keeps the
deployed image small enough to cold-start fast.

Everything is written to disk first. The upload is best-effort — if the network
or the service is down we keep collecting, and nothing is lost.

  python collect.py                      # collect + save locally
  INGEST_URL=https://…/api/ingest python collect.py    # also stream to cloud
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import platform
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(os.environ.get("OUT_DIR", ROOT / "collector" / "data"))
FRAMES = OUT / "frames"
SWEEPS = OUT / "sweeps.jsonl"

CYCLE_SECONDS = int(os.environ.get("CYCLE_SECONDS", "240"))
FETCH_WORKERS = int(os.environ.get("FETCH_WORKERS", "6"))
FETCH_COOLDOWN = float(os.environ.get("FETCH_COOLDOWN", "0.35"))
SAVE_FRAMES = os.environ.get("SAVE_FRAMES", "1") == "1"
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "1"))     # keep frames every Nth sweep
# Frames are 352x240, so a pedestrian mid-block is ~5px and a car is ~10x the
# pixel area. Upscaling to 640 helps; dropping confidence does not help equally
# for every class. At 0.15 the model starts calling orange traffic cones
# "person" (verified by drawing boxes on real frames), so person gets a higher
# bar than vehicles. Counts are for relative change, not census.
IMGSZ = int(os.environ.get("DETECTOR_IMGSZ", "640"))
CONF = float(os.environ.get("DETECTOR_CONF", "0.15"))     # floor for predict()
CLASS_CONF = {
    "person": float(os.environ.get("CONF_PERSON", "0.35")),
    "bicycle": float(os.environ.get("CONF_BICYCLE", "0.30")),
    "car": float(os.environ.get("CONF_CAR", "0.20")),
    "motorcycle": float(os.environ.get("CONF_MOTORCYCLE", "0.25")),
    "bus": float(os.environ.get("CONF_BUS", "0.25")),
    "truck": float(os.environ.get("CONF_TRUCK", "0.25")),
}
# A cone is roughly as wide as it is tall; a standing person is not.
MIN_PERSON_ASPECT = float(os.environ.get("MIN_PERSON_ASPECT", "1.35"))
DETECT_BATCH = int(os.environ.get("DETECT_BATCH", "16"))
INGEST_URL = os.environ.get("INGEST_URL", "")
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")

CLASS_MAP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COUNT_KEYS = list(dict.fromkeys(CLASS_MAP.values()))

CAMS: List[Dict] = json.loads(
    (ROOT / "camera_direction" / "data" / "manhattan_cams.json").read_text(encoding="utf-8-sig")
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- model ----------------------------------------------------------------

def build_model():
    import torch

    # ultralytics 8.2.x predates torch>=2.6's weights_only default; the file is
    # the official ultralytics release asset.
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

    from ultralytics import YOLO

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    m = YOLO(os.environ.get("YOLO_WEIGHTS", "yolov8m.pt"))
    m.to(dev)
    name = torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor() or "cpu"
    log(f"model on {dev} ({name})")
    return m, dev, name


# --- fetch ----------------------------------------------------------------

def fetch(cam: Dict) -> Optional[bytes]:
    """One frame, politely. The cooldown keeps us a good citizen on a public
    municipal endpoint; being throttled mid-demo is worse than a slow sweep."""
    try:
        r = requests.get(cam["url"], timeout=15)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    finally:
        time.sleep(FETCH_COOLDOWN)
    return None


def sweep(model, dev: str, dev_name: str, n: int) -> Dict:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        frames = list(ex.map(fetch, CAMS))
    got = [i for i, f in enumerate(frames) if f]
    fetch_s = time.time() - t0

    ts = time.time()
    stamp = int(ts)

    if SAVE_FRAMES and n % SAVE_EVERY == 0:
        d = FRAMES / str(stamp)
        d.mkdir(parents=True, exist_ok=True)
        for i in got:
            try:
                (d / f"{CAMS[i]['id']}.jpg").write_bytes(frames[i])
            except Exception:
                pass

    from PIL import Image
    imgs, keep = [], []
    for i in got:
        try:
            imgs.append(Image.open(io.BytesIO(frames[i])).convert("RGB"))
            keep.append(i)
        except Exception:
            pass

    t1 = time.time()
    counts: Dict[str, Dict[str, int]] = {}
    totals = {k: 0 for k in COUNT_KEYS}

    # Chunked: handing all ~370 frames to predict() at once asks for ~7GB of
    # VRAM on the medium model and OOMs an 8GB card. Batching costs nothing at
    # a multi-minute cycle.
    pos = 0
    while pos < len(imgs):
        chunk = imgs[pos:pos + DETECT_BATCH]
        for i, res in zip(keep[pos:pos + DETECT_BATCH],
                          model.predict(chunk, verbose=False, imgsz=IMGSZ,
                                        conf=CONF, device=dev)):
            c = {k: 0 for k in COUNT_KEYS}
            boxes = res.boxes
            for cls, score, xyxy in zip(boxes.cls.tolist(),
                                        boxes.conf.tolist(),
                                        boxes.xyxy.tolist()):
                lbl = CLASS_MAP.get(int(cls))
                if not lbl or score < CLASS_CONF[lbl]:
                    continue
                if lbl == "person":
                    w = max(xyxy[2] - xyxy[0], 1e-6)
                    if (xyxy[3] - xyxy[1]) / w < MIN_PERSON_ASPECT:
                        continue      # too squat to be a standing person
                c[lbl] += 1
                totals[lbl] += 1
            counts[CAMS[i]["id"]] = c
        pos += DETECT_BATCH

    if dev == "cuda":
        import torch
        torch.cuda.empty_cache()
    detect_s = time.time() - t1

    return {
        "ts": ts, "ok": len(keep), "attempted": len(CAMS),
        "fetch_s": round(fetch_s, 1), "detect_s": round(detect_s, 1),
        "device": f"{dev}:{dev_name}", "totals": totals, "counts": counts,
    }


def push(cycle: Dict) -> str:
    if not INGEST_URL:
        return "local-only"
    try:
        r = requests.post(INGEST_URL, json=cycle, timeout=30,
                          headers={"X-Ingest-Token": INGEST_TOKEN})
        return "pushed" if r.status_code == 200 else f"push {r.status_code}"
    except Exception as e:
        return f"push failed ({type(e).__name__})"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"{len(CAMS)} Manhattan cameras · cycle {CYCLE_SECONDS}s · "
        f"{FETCH_WORKERS} workers · {FETCH_COOLDOWN}s cooldown")
    log(f"writing to {OUT}")
    log(f"ingest: {INGEST_URL or 'not configured (saving locally only)'}")

    model, dev, dev_name = build_model()
    n = 0
    while True:
        start = time.time()
        try:
            cyc = sweep(model, dev, dev_name, n)
            with SWEEPS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(cyc) + "\n")
            state = push(cyc)
            t = cyc["totals"]
            log(f"#{n:<4d} {cyc['ok']:>3d}/{cyc['attempted']} cams · "
                f"fetch {cyc['fetch_s']:>4.1f}s detect {cyc['detect_s']:>4.1f}s · "
                f"people {t['person']:>4d} bikes {t['bicycle']:>3d} "
                f"cars {t['car']:>4d} · {state}")
            n += 1
        except KeyboardInterrupt:
            log("stopped")
            sys.exit(0)
        except Exception as e:
            log(f"sweep failed: {type(e).__name__}: {e}")
        time.sleep(max(5.0, CYCLE_SECONDS - (time.time() - start)))


if __name__ == "__main__":
    main()
