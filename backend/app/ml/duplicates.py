"""
Duplicate detection: perceptual image hash (phash) + geo-proximity + same
category. This runs entirely offline with no heavy model download, which
keeps the sandbox/demo fast to boot.

UPGRADE PATH (per the dev guide): swap `imagehash.phash` for CLIP image
embeddings + cosine similarity if you want robustness to very different
camera angles of the same pothole:

    import torch, clip
    _clip_model, _clip_preprocess = clip.load("ViT-B/32")

    def get_embedding(image_path):
        img = _clip_preprocess(Image.open(image_path)).unsqueeze(0)
        with torch.no_grad():
            return _clip_model.encode_image(img).numpy()

    # then compare cosine similarity > 0.9 instead of hash distance below.
"""

import imagehash
from PIL import Image
from math import radians, sin, cos, sqrt, atan2

DUPLICATE_RADIUS_METERS = 100
HASH_DISTANCE_THRESHOLD = 10  # lower = stricter match (phash hamming distance, 0-64)


def compute_phash(image_path: str) -> str:
    return str(imagehash.phash(Image.open(image_path).convert("RGB")))


def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def find_duplicate(db_reports, new_phash: str, category: str, lat: float, lng: float):
    """db_reports: iterable of Report ORM objects already in the DB.
    Returns the matching Report or None."""
    new_hash_obj = imagehash.hex_to_hash(new_phash)
    for r in db_reports:
        if r.category != category or not r.phash:
            continue
        dist_m = haversine_meters(lat, lng, r.latitude, r.longitude)
        if dist_m > DUPLICATE_RADIUS_METERS:
            continue
        hash_dist = new_hash_obj - imagehash.hex_to_hash(r.phash)
        if hash_dist <= HASH_DISTANCE_THRESHOLD:
            return r
    return None
