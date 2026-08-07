"""Re-score every saved frame at different person thresholds.

Non-destructive by design: reads only the saved JPEGs, writes to its own
output files, and never touches sweeps.jsonl or the running collector. That
means the live count data stays exactly as collected while we evaluate
whether a looser threshold is actually better.

Detection runs once per frame at a low floor; the class thresholds are applied
afterwards, so several variants come out of a single GPU pass.

  python reprocess.py                 # writes sweeps_p20.jsonl / sweeps_p25.jsonl
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
FRAMES = DATA / "frames"

CLASS_MAP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COUNT_KEYS = list(dict.fromkeys(CLASS_MAP.values()))

FLOOR = 0.12                       # detect once this low, threshold later
ASPECT = float(os.environ.get("MIN_PERSON_ASPECT", "1.35"))
IMGSZ = int(os.environ.get("DETECTOR_IMGSZ", "640"))
BATCH = int(os.environ.get("DETECT_BATCH", "8"))    # small: collector shares the GPU

# person threshold -> output file
VARIANTS = {0.20: DATA / "sweeps_p20.jsonl",
            0.25: DATA / "sweeps_p25.jsonl"}
OTHER_CONF = {"bicycle": 0.30, "car": 0.20, "motorcycle": 0.25,
              "bus": 0.25, "truck": 0.25}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_model():
    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = YOLO(os.environ.get("YOLO_WEIGHTS", "yolov8m.pt"))
    m.to(dev)
    log(f"model on {dev}")
    return m, dev


def main() -> None:
    import torch
    from PIL import Image

    cycles = sorted(FRAMES.iterdir(), key=lambda p: int(p.name))
    total = sum(1 for c in cycles for _ in c.glob("*.jpg"))
    log(f"{len(cycles)} sweeps, {total} frames — writing {', '.join(p.name for p in VARIANTS.values())}")
    log("sweeps.jsonl is NOT touched; live collector keeps running")

    model, dev = build_model()
    handles = {t: p.open("w", encoding="utf-8") for t, p in VARIANTS.items()}
    running = {t: {k: 0 for k in COUNT_KEYS} for t in VARIANTS}
    baseline_people = 0
    t_start = time.time()

    for n, cyc in enumerate(cycles):
        paths = sorted(cyc.glob("*.jpg"))
        if not paths:
            continue
        ts = float(cyc.name)

        per_cam = {t: {} for t in VARIANTS}
        totals = {t: {k: 0 for k in COUNT_KEYS} for t in VARIANTS}

        for i in range(0, len(paths), BATCH):
            chunk = paths[i:i + BATCH]
            imgs, ids = [], []
            for p in chunk:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    ids.append(p.stem)
                except Exception:
                    pass
            if not imgs:
                continue
            try:
                results = model.predict(imgs, verbose=False, imgsz=IMGSZ,
                                        conf=FLOOR, device=dev)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                time.sleep(2)
                results = model.predict(imgs, verbose=False, imgsz=IMGSZ,
                                        conf=FLOOR, device="cpu")

            for cam_id, res in zip(ids, results):
                b = res.boxes
                dets = list(zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist()))
                for thr in VARIANTS:
                    c = {k: 0 for k in COUNT_KEYS}
                    for cls, sc, xy in dets:
                        lbl = CLASS_MAP.get(int(cls))
                        if not lbl:
                            continue
                        if lbl == "person":
                            if sc < thr:
                                continue
                            w = max(xy[2] - xy[0], 1e-6)
                            if (xy[3] - xy[1]) / w < ASPECT:
                                continue
                        elif sc < OTHER_CONF[lbl]:
                            continue
                        c[lbl] += 1
                        totals[thr][lbl] += 1
                    per_cam[thr][cam_id] = c

        for thr, fh in handles.items():
            fh.write(json.dumps({
                "ts": ts, "ok": len(per_cam[thr]), "attempted": len(paths),
                "fetch_s": 0, "detect_s": 0,
                "device": f"reprocessed@person{thr}",
                "totals": totals[thr], "counts": per_cam[thr],
            }) + "\n")
            fh.flush()
            for k, v in totals[thr].items():
                running[thr][k] += v

        log(f"  sweep {n+1}/{len(cycles)}  {time.strftime('%H:%M', time.localtime(ts))}  "
            + "  ".join(f"p{int(t*100)}={totals[t]['person']:>4d}" for t in VARIANTS))

    for fh in handles.values():
        fh.close()

    log(f"done in {time.time()-t_start:.0f}s")
    for thr in VARIANTS:
        log(f"  person>={thr}: {running[thr]['person']} people, "
            f"{running[thr]['car']} cars across all sweeps")


if __name__ == "__main__":
    main()
