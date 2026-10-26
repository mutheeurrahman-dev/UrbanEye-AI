"""
UrbanEye AI — inference module.

WHY A HEURISTIC MODEL LIVES HERE RIGHT NOW:
This sandbox has no GPU and no labeled civic-issue dataset, so there is no
honest way to train/ship a real YOLOv8 detector here. Per the dev guide's own
strategy ("Backend engineer can build and test all of this against MOCKED ML
outputs first, then swap in the real model once inference is ready"), this
module implements that exact contract: same function signature, same return
shape, so your ML engineer can drop in the real trained model with a
one-function change and nothing else in the codebase needs to move.

HOW TO SWAP IN THE REAL MODEL (once you've trained YOLOv8n per the guide):

    from ultralytics import YOLO
    _model = YOLO("runs/detect/train/weights/best.onnx")

    def run_inference(image_path: str) -> dict:
        results = _model(image_path)[0]
        if len(results.boxes) == 0:
            return {"category": "unknown", "confidence": 0.0, "box_area_ratio": 0.0}
        box = max(results.boxes, key=lambda b: b.conf)
        cls_name = results.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        img_w, img_h = results.orig_shape[1], results.orig_shape[0]
        box_area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        return {"category": cls_name, "confidence": conf, "box_area_ratio": box_area_ratio}

That's it — the rest of the API (severity scoring, routing, duplicate
detection, dashboard) already consumes exactly this dict shape and needs no
other changes.
"""

import hashlib
from PIL import Image
import numpy as np

from app.models import CLASS_LIST

CONFIDENCE_THRESHOLD = 0.55  # below this -> needs_review, per the guide's "responsible AI" fallback


def _image_color_signature(image_path: str) -> dict:
    """Cheap, real (non-random) signal from the actual uploaded pixels:
    average hue/brightness/saturation, used to pick a plausible class so the
    demo reacts to the actual photo rather than being pure noise."""
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.asarray(img).astype(float) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = float((r + g + b).mean() / 3)
    blueness = float((b - (r + g) / 2).mean())
    greenness = float((g - (r + b) / 2).mean())
    grayness = float(1 - np.abs(r - g).mean() - np.abs(g - b).mean())
    edges = float(np.abs(np.diff(arr.mean(axis=2), axis=0)).mean() +
                   np.abs(np.diff(arr.mean(axis=2), axis=1)).mean())
    return dict(brightness=brightness, blueness=blueness, greenness=greenness,
                grayness=grayness, edges=edges)


def run_inference(image_path: str) -> dict:
    """Placeholder inference — deterministic per-image, swappable per the
    docstring above. Returns {category, confidence, box_area_ratio}."""
    sig = _image_color_signature(image_path)

    scores = {
        "waterlogging": sig["blueness"] * 3 + (1 - sig["edges"]),
        "garbage": sig["greenness"] * 2 + sig["edges"] * 0.5,
        "pothole": sig["grayness"] * 1.5 + sig["edges"] * 1.2 - sig["brightness"],
        "road_damage": sig["grayness"] * 1.2 + sig["edges"] * 0.8,
        "crack": sig["grayness"] * 1.0 + sig["edges"] * 1.5 - sig["brightness"] * 0.5,
        "broken_light": (1 - sig["brightness"]) * 1.3,
    }
    category = max(scores, key=scores.get)

    # deterministic pseudo-confidence + box_area_ratio from image hash, so
    # repeat-uploads of the identical photo always score identically
    h = int(hashlib.sha256(open(image_path, "rb").read()).hexdigest(), 16)
    confidence = 0.55 + (h % 4000) / 10000  # 0.55 - 0.95
    box_area_ratio = 0.10 + (h % 6000) / 10000  # 0.10 - 0.70

    return {
        "category": category,
        "confidence": round(confidence, 3),
        "box_area_ratio": round(box_area_ratio, 3),
    }


def severity_score(box_area_ratio: float, cls_name: str) -> float:
    base = {
        "pothole": 0.6, "crack": 0.7, "road_damage": 0.5,
        "waterlogging": 0.5, "garbage": 0.3, "broken_light": 0.4,
    }.get(cls_name, 0.4)
    return round(min(1.0, base + box_area_ratio * 0.5), 2)
