# 👁 Naina

**Cameras that know which way they're looking.**

**Live:** https://nyc-vision-agent-978609869472.us-central1.run.app

NYC DOT publishes ~940 live traffic cameras. It does **not** publish which
direction any of them point. That one missing field is the difference between
finding the camera *nearest* a problem and finding the camera that can
*actually see* it — and everything here is built on closing that gap.

We recovered headings for **234 of Manhattan's 373 cameras**, then used them
for two things: a live map of where people and vehicles are across the whole
borough, and an agent that checks whether a 311 complaint is still true.

---

## What it does

### 1. Direction-aware 311 triage

NYC takes ~3 million 311 complaints a year. Someone reports *"car blocking my
driveway"* — and by the time an inspector arrives, hours or days later, the car
is usually gone. The city spends inspection capacity on complaints that already
resolved themselves.

Naina takes a complaint, computes the bearing from each nearby camera to the
incident, compares it against that camera's known heading, and keeps only the
ones actually pointed at it. Then Gemini looks at the live frame and rules.

```
Blocked Driveway — 158 E 126 St
  → Lexington Ave @ E 128 St · 173m · facing SW · 35° off-axis · IN VIEW
  → not_visible (90%)
    "No vehicles are seen blocking any driveways or curb cuts along the street."
```

A camera 40m away facing the wrong way is useless. One 300m down the block
pointed straight at it is not. Proximity alone cannot tell those apart.

### 2. Borough-wide activity timeline

Every 4 minutes we sweep all 373 Manhattan cameras, count people and vehicles
in each frame, and store the sweep. The UI plots it live with per-class toggles
and a **Play** control that replays the evening.

This is *correlated density over time*, not individual tracking. These cameras
are 352×240 and minutes apart; re-identifying a person across them is not
possible and we don't claim it.

---

## Architecture

```
   NYC DOT cameras  ──┐
   (373 Manhattan)    │
                      ▼
             ┌──────────────────┐   counts only    ┌─────────────────────┐
             │  edge collector  │ ───── POST ────▶ │  Cloud Run: Naina   │
             │  local GPU       │   /api/ingest    │                     │
             │  YOLOv8m · ~6s   │    (~40KB)       │  · map + timeline   │
             └──────────────────┘                  │  · 311 agent        │
                      │                            │  · Gemini verdict   │
                      ▼                            └─────────────────────┘
           frames + sweeps to disk                          │
                                                    NYC Open Data 311
```

**Why detection runs at the edge.** An RTX 5070 does all 373 cameras in ~6
seconds; Cloud Run's CPU needs ~12 and would need torch in the image (~2.5GB).
Keeping inference local means the container is tens of MB and cold-starts fast.
Only counts cross the wire — ~40KB per sweep, no imagery.

**The Cloud Run service is not a thin shell.** It owns the direction geometry,
the 311 agent, the Gemini verification, and the UI. If the collector stops,
verification keeps working and the timeline still replays.

**Auth is ADC, not API keys.** The lab project disallows API keys by policy, so
the container authenticates as its own service account through Vertex AI. There
is no key in the repo or the deploy.

---

## Things we got wrong, and how we found out

**The model was counting traffic cones as people.** Pedestrian counts looked
low, so we lowered the confidence threshold and counts tripled — which we
initially read as success. Drawing the bounding boxes showed **71% of the new
"people" were orange traffic cones.** Fixed with per-class thresholds plus an
aspect-ratio filter: a standing person is taller than wide, a cone isn't.

**The counts were internally inconsistent.** We tuned detection three times
during the evening, so the timeline had step-changes that were tuning
artefacts, not real activity. Because every frame is written to disk before
upload, we **reprocessed all 11,439 saved frames** in 124 seconds at a single
threshold, giving a uniform series end to end.

**Gemini answered "inconclusive" to almost everything.** It was being asked to
verify a complaint *at a street address*, which no 352×240 frame can resolve.
Restructured to a two-step judgement — rate whether the frame is usable, then
rule on the condition — with dark/wet/grainy explicitly flagged as normal
rather than disqualifying. Measured on 12 live complaints: **7 of 8 retrieved
frames now reach a verdict at 0.85–0.95 confidence.**

---

## Honest limitations

- **Cars are reliable; people are directional.** At 352×240 a pedestrian
  mid-block is ~5 pixels. It rained all evening. Treat person counts as
  relative change, not a census.
- **No tracking.** Density over time, never trajectories.
- **Headings are 8-point**, so "in view" is a ±55° cone — enough to exclude a
  camera facing the wrong way, not enough to guarantee a specific storefront.
- **139 cameras have no heading.** The API reports `dir: null` and they're
  excluded from "in view" rather than guessed.
- **311 publishes with ~43 hours of lag.** The cameras are live; the complaints
  are recent, not live.
- **~1 in 3 camera fetches fail.** We try up to four cameras twice each, and it
  still sometimes returns no frame.

---

## Where the headings came from

- **OCR (150 cameras)** — many DOT cameras burn a banner into the frame naming
  the location and direction. Read it directly.
- **DINO embedding match (84 cameras)** — match the live frame against Street
  View panoramas at known bearings; best match is the heading. Scores 0.55–0.86.

Spot-checks hold: the data says 1 Ave @ 96 St faces **N** and 2 Ave @ 36 St
faces **S** — the true one-way directions of those avenues.

---

## Running it

**Cloud Run**
```bash
gcloud run deploy nyc-vision-agent --source . --region us-central1 \
  --allow-unauthenticated --memory 512Mi --min-instances 1 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,\
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,GOOGLE_CLOUD_LOCATION=global,\
INGEST_TOKEN=YOUR_TOKEN
```

**Edge collector**
```bash
cd collector
INGEST_URL=https://YOUR-SERVICE.run.app/api/ingest \
INGEST_TOKEN=YOUR_TOKEN python collect.py
```

**Reprocess saved frames** (non-destructive; writes its own files)
```bash
cd collector && python reprocess.py
```

## Endpoints

| Route | What it gives you |
|---|---|
| `GET /` | map + timeline UI |
| `GET /api/health` | sweeps, edge device, heading coverage, resolved model |
| `GET /api/cameras` | 373 cameras with headings |
| `GET /api/cycles?since=` | sweep history, incremental |
| `GET /api/311?days=&limit=` | complaints with cameras that can see them |
| `POST /api/verify` | direction-aware camera pick + Gemini verdict |
| `POST /api/ingest` | collector upload (token-gated) |

## Stack

FastAPI on Cloud Run · Gemini 3.5 Flash via Vertex AI + ADC · YOLOv8m at the
edge · Leaflet + CARTO · NYC Open Data (`erm2-nwe9`) · NYCTMC camera API
