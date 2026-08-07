# NYC Vision Agent

**Cameras that know which way they're looking.**

NYC DOT publishes ~940 live traffic cameras. It does *not* publish which
direction any of them point. That missing field is the difference between
finding the camera **nearest** to a problem and finding the camera that can
**actually see** it — and it's what this project is built on.

We recovered headings for 234 of Manhattan's 373 cameras, then used them for
two things: a live map of where people, bikes and cars are across the whole
borough, and an agent that checks whether a 311 complaint is still visible on
the street right now.

---

## What it does

**1. Direction-aware 311 verification.** A complaint comes in at a lat/lon. We
find cameras within range, compute the bearing from each camera to the
incident, compare it against that camera's known heading, and keep only the
ones actually pointed at it. Then Gemini looks at the live frame and judges
whether the reported condition is visible.

A camera 40m away pointed the wrong way is useless. A camera 300m away pointed
straight down the block is not. Proximity alone can't tell those apart.

**2. Borough-wide activity map.** Every ~90 seconds we sweep all 373 Manhattan
cameras, count people / bicycles / cars in each frame, and store the sweep.
The UI plots it live, with per-class toggles and a scrubber to replay the
evening. Density shifts along a corridor read as movement — but this is
*correlated density over time*, not individual tracking, and we don't claim
otherwise. These cameras are 352×240 and seconds apart; re-identifying a person
across them isn't possible and we didn't pretend it was.

---

## Architecture

```
   NYC DOT cameras  ──┐
   (373 Manhattan)    │
                      ▼
             ┌──────────────────┐   counts only    ┌─────────────────────┐
             │  edge collector  │ ───── POST ────▶ │  Cloud Run service  │
             │  (local, GPU)    │   /api/ingest    │                     │
             │  YOLOv8n · ~2s   │                  │  · live map UI      │
             └──────────────────┘                  │  · 311 agent        │
                      │                            │  · Gemini verify    │
                      ▼                            └─────────────────────┘
              frames + sweeps                                │
              saved to disk                          NYC Open Data 311
```

**Why detection runs at the edge.** An RTX 5070 does all 373 cameras in ~2
seconds; Cloud Run's CPU needs ~12. Keeping torch out of the container also
drops the image from ~2.5GB to tens of MB, so cold starts are quick. Only
counts cross the wire — a few KB per sweep, no imagery.

**The Cloud Run service is not a thin shell.** It owns the 311 agent, the
direction geometry, the Gemini verification, and the UI. If the edge collector
disappears mid-demo, verification keeps working and the map still replays
everything collected so far.

---

## Running it

### Cloud Run

```bash
gcloud run deploy nyc-vision-agent \
  --source . --region us-central1 --allow-unauthenticated \
  --memory 512Mi --min-instances 1 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,\
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,GOOGLE_CLOUD_LOCATION=us-central1,\
INGEST_TOKEN=YOUR_TOKEN
```

Auth is Application Default Credentials — the container authenticates as its
own service account, so there is no API key anywhere in the repo or the
deploy. (The lab project disallows API keys by org policy, which is the
better pattern regardless.)

### Edge collector

```bash
cd collector
INGEST_URL=https://YOUR-SERVICE.run.app/api/ingest \
INGEST_TOKEN=YOUR_TOKEN \
python collect.py
```

Writes every frame and sweep to `collector/data/` first, then uploads.
Best-effort: if the network or the service is down, collection continues and
nothing is lost.

---

## Endpoints

| Route | What it gives you |
|---|---|
| `GET /` | Live map UI |
| `GET /healthz` | Sweep count, edge device, camera/heading coverage |
| `GET /api/cameras` | All 373 cameras with headings |
| `GET /api/latest` | Most recent sweep |
| `GET /api/series?camera_id=` | Time series, borough-wide or per camera |
| `GET /api/311?limit=` | Recent Manhattan complaints, each with cameras that can see it |
| `POST /api/verify` | Direction-aware camera pick + Gemini verdict |
| `POST /api/ingest` | Edge collector sweep upload (token-gated) |

---

## Where the headings came from

Two methods, in `camera_direction/`:

- **OCR (150 cameras)** — many DOT cameras burn a text banner into the frame
  naming the location and direction. Read it directly.
- **DINO embedding match (84 cameras)** — match the live frame against Google
  Street View panoramas taken at the camera's coordinates at known bearings;
  the best-matching bearing is the camera's heading. Scores 0.55–0.86.

Spot-checks hold up against ground truth: the data says 1 Ave @ 96 St faces
**N** and 2 Ave @ 36 St faces **S**, which are the true one-way directions of
those avenues.

139 Manhattan cameras still have no heading. The API reports `dir: null` for
them and the 311 agent excludes them from "in view" rather than guessing.

---

## Honest limitations

- **No individual tracking.** Density over time, not trajectories. See above.
- **Counts are YOLOv8n at 352×240.** Reliable for relative change and busy /
  quiet; not a census. Occlusion and night frames undercount.
- **Headings are 8-point** (N/NE/E/…), so "in view" is a ±55° cone, not a
  precise frustum. Good enough to exclude a camera facing the opposite way,
  not good enough to guarantee a specific storefront is in shot.
- **311 lag.** Complaints are filed minutes to hours after the fact, so
  "not visible" often means "already cleared," which is itself useful.

---

## Stack

FastAPI on Cloud Run · Gemini 3.5 Flash via Vertex AI + ADC · YOLOv8n
(ultralytics) at the edge · Leaflet + CARTO dark tiles · NYC Open Data
(`erm2-nwe9`) · NYCTMC camera API
