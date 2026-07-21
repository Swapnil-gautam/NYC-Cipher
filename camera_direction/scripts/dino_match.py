import os
import json
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

PROJ = r"e:\Projects\NYC_Data\camera_direction"
SV_TEST_DIR = os.path.join(PROJ, "images", "street_view_test")
OUT_DIR = os.path.join(PROJ, "results", "composites_vith16plus")
os.makedirs(OUT_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]
HEADING_LABEL = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}

print(f"Device: {DEVICE}")
print("Loading model (facebook/dinov3-vith16plus-pretrain-lvd1689m, ~3.2GB)...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModel.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE).eval()

def embed(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    cls = out.last_hidden_state[:, 0]  # CLS token as global embedding
    return F.normalize(cls, dim=-1)[0].cpu()

def make_composite(frame_path, best_path, best_heading, best_score, out_path, cam_name):
    frame_img = Image.open(frame_path).convert("RGB").resize((352, 240))
    best_img = Image.open(best_path).convert("RGB").resize((352, 240))
    pad = 10
    label_h = 26
    W = frame_img.width + best_img.width + pad * 3
    H = frame_img.height + label_h + pad * 2
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 4), f"{cam_name}", fill=(255, 255, 255))
    y = label_h + pad
    canvas.paste(frame_img, (pad, y))
    canvas.paste(best_img, (pad * 2 + frame_img.width, y))
    draw.text((pad, y + frame_img.height + 2), "CAMERA FRAME", fill=(150, 200, 255))
    draw.text((pad * 2 + frame_img.width, y + frame_img.height + 2),
              f"BEST MATCH: heading {best_heading} ({HEADING_LABEL[best_heading]}) sim={best_score:.3f}",
              fill=(150, 255, 150))
    canvas.save(out_path)

results = []
cam_dirs = sorted([d for d in os.listdir(SV_TEST_DIR) if os.path.isdir(os.path.join(SV_TEST_DIR, d))])
print(f"Found {len(cam_dirs)} camera dirs: {cam_dirs}")

for cam_name in cam_dirs:
    cdir = os.path.join(SV_TEST_DIR, cam_name)
    frame_path = os.path.join(cdir, "frame.jpg")
    if not os.path.exists(frame_path):
        continue
    frame_emb = embed(frame_path)

    scores = {}
    for h in HEADINGS:
        sv_path = os.path.join(cdir, f"sv_{h}.jpg")
        if not os.path.exists(sv_path):
            continue
        sv_emb = embed(sv_path)
        sim = torch.dot(frame_emb, sv_emb).item()
        scores[h] = sim

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_h, best_s = ranked[0]
    result = {
        "name": cam_name,
        "best_heading": best_h,
        "best_label": HEADING_LABEL[best_h],
        "best_score": best_s,
        "all_scores": {HEADING_LABEL[k]: v for k, v in ranked}
    }
    results.append(result)

    out_composite = os.path.join(OUT_DIR, f"{cam_name}.jpg")
    make_composite(frame_path, os.path.join(cdir, f"sv_{best_h}.jpg"), best_h, best_s, out_composite, cam_name)
    print(f"{cam_name}: best={best_h} ({HEADING_LABEL[best_h]}) sim={best_s:.3f} | all: " +
          ", ".join(f"{k}={v:.3f}" for k, v in ranked))

out_json = os.path.join(PROJ, "results", "dino_results_vith16plus.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_json}")
print("DONE")
