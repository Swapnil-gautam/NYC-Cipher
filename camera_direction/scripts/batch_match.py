import os
import re
import sys
import json
import time
import urllib.request
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

PROJ = r"e:\Projects\NYC_Data\camera_direction"
IMG_DIR = os.path.join(PROJ, "images", "street_view_full")
COMPOSITE_DIR = os.path.join(PROJ, "results", "composites_full")
RESULTS_PATH = os.path.join(PROJ, "results", "batch_results.json")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(COMPOSITE_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
MAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]
HEADING_LABEL = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
CONFIDENCE_THRESHOLD = 0.55

CAM_FRAME_COOLDOWN = 0.6   # seconds between NYCTMC live-frame fetches (this endpoint choked last time)
SV_COOLDOWN = 0.15         # seconds between Street View fetches (much higher limit, light pacing only)

def slugify(name):
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:60]

def fetch(url, out_path, retries=2):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt == retries:
                print(f"  FETCH FAILED ({url[:60]}...): {e}")
                return False
            time.sleep(1.0)
    return False

print(f"Device: {DEVICE}")
print("Loading model...")
t0 = time.time()
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModel.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE).eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

def embed(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    cls = out.last_hidden_state[:, 0]
    return F.normalize(cls, dim=-1)[0].cpu()

def make_composite(frame_path, best_path, best_heading, best_score, out_path, cam_name, confident):
    frame_img = Image.open(frame_path).convert("RGB").resize((352, 240))
    best_img = Image.open(best_path).convert("RGB").resize((352, 240))
    pad = 10
    label_h = 26
    W = frame_img.width + best_img.width + pad * 3
    H = frame_img.height + label_h + pad * 2
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    tag = "CONFIDENT" if confident else "LOW CONFIDENCE - SKIPPED"
    draw.text((pad, 4), f"{cam_name}  [{tag}]", fill=(255, 255, 255) if confident else (255, 150, 150))
    y = label_h + pad
    canvas.paste(frame_img, (pad, y))
    canvas.paste(best_img, (pad * 2 + frame_img.width, y))
    draw.text((pad, y + frame_img.height + 2), "CAMERA FRAME", fill=(150, 200, 255))
    draw.text((pad * 2 + frame_img.width, y + frame_img.height + 2),
              f"heading {best_heading} ({HEADING_LABEL[best_heading]}) sim={best_score:.3f}",
              fill=(150, 255, 150) if confident else (255, 180, 120))
    canvas.save(out_path)

def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else start_idx + 10

    with open(os.path.join(PROJ, "data", "manhattan_cams.json"), encoding="utf-8-sig") as f:
        all_cams = json.load(f)
    with open(os.path.join(PROJ, "data", "manhattan_directions.json"), encoding="utf-8-sig") as f:
        ocr_results = json.load(f)

    no_dir_ids = {r["id"] for r in ocr_results if not r.get("direction")}
    candidates = [c for c in all_cams if c["id"] in no_dir_ids]
    print(f"Total candidates (no OCR direction): {len(candidates)}")

    batch = candidates[start_idx:end_idx]
    print(f"Processing indices [{start_idx}:{end_idx}] -> {len(batch)} cameras")

    results = load_existing_results()
    done_ids = {r["id"] for r in results}

    t_batch_start = time.time()
    for i, cam in enumerate(batch):
        if cam["id"] in done_ids:
            print(f"[{start_idx+i}] {cam['name']}: already done, skipping")
            continue

        t0 = time.time()
        slug = slugify(cam["name"]) + "_" + cam["id"][:6]
        cdir = os.path.join(IMG_DIR, slug)
        os.makedirs(cdir, exist_ok=True)

        frame_path = os.path.join(cdir, "frame.jpg")
        ok = fetch(cam["url"], frame_path)
        time.sleep(CAM_FRAME_COOLDOWN)
        if not ok:
            print(f"[{start_idx+i}] {cam['name']}: FRAME FETCH FAILED, skipping")
            results.append({"id": cam["id"], "name": cam["name"], "slug": slug, "status": "frame_fetch_failed"})
            save_results(results)
            continue

        sv_paths = {}
        for h in HEADINGS:
            sv_url = f"https://maps.googleapis.com/maps/api/streetview?size=300x200&location={cam['latitude']},{cam['longitude']}&heading={h}&fov=90&pitch=0&key={MAPS_KEY}"
            sv_path = os.path.join(cdir, f"sv_{h}.jpg")
            if fetch(sv_url, sv_path):
                sv_paths[h] = sv_path
            time.sleep(SV_COOLDOWN)

        if not sv_paths:
            print(f"[{start_idx+i}] {cam['name']}: NO STREET VIEW IMAGES, skipping")
            results.append({"id": cam["id"], "name": cam["name"], "slug": slug, "status": "no_street_view"})
            save_results(results)
            continue

        frame_emb = embed(frame_path)
        scores = {h: torch.dot(frame_emb, embed(p)).item() for h, p in sv_paths.items()}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_h, best_s = ranked[0]
        confident = best_s >= CONFIDENCE_THRESHOLD

        result = {
            "id": cam["id"],
            "name": cam["name"],
            "slug": slug,
            "status": "confident" if confident else "low_confidence",
            "best_heading": best_h,
            "best_label": HEADING_LABEL[best_h],
            "best_score": best_s,
            "all_scores": {HEADING_LABEL[k]: v for k, v in ranked}
        }
        results.append(result)
        save_results(results)

        make_composite(frame_path, sv_paths[best_h], best_h, best_s,
                        os.path.join(COMPOSITE_DIR, f"{slug}.jpg"), cam["name"], confident)

        elapsed = time.time() - t0
        tag = "OK" if confident else "WEAK"
        print(f"[{start_idx+i}] {cam['name']}: {tag} best={HEADING_LABEL[best_h]} sim={best_s:.3f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_batch_start
    n = len(batch)
    print(f"\nBatch done: {n} cameras in {total_elapsed:.1f}s ({total_elapsed/max(n,1):.1f}s/camera avg)")

if __name__ == "__main__":
    main()
