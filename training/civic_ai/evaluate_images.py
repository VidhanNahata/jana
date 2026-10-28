"""
Measure the local VLM's image understanding (stage 1) on labelled photos.

    python training/civic_ai/evaluate_images.py --n 40

Positives: pothole test images (training/data/potholes/test/images, held out
from detector training). Negatives: real non-civic photos (GenImage "real"
images = ImageNet photos, training/data/authenticity/test/real; a few ImageNet
classes such as street signs may show public scenes). Only the VLM runs (no
pothole detector), so this measures the VLM alone.

Reports: valid-JSON rate (after the one retry), pothole recall (ROAD.POTHOLE
listed), road-problem recall (any ROAD.* listed), non-civic specificity
(is_civic_scene=false and no problem listed), false civic-problem rate on
negatives, latency per image, peak process memory and accelerator memory.
Note: some ImageNet photos are street scenes, so is_civic_scene=true on a
"negative" is not necessarily wrong; the false civic-problem rate is the
meaningful negative metric.
No local labelled photos exist for other categories (garbage, drains...), so
those are not measured here.
"""

import argparse
import json
import os
import resource
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ["AI_VLM_ENABLED"] = "true"

from backend.ai import vision  # noqa: E402
from backend.ai.schema import StageStatus  # noqa: E402

POS = os.path.join(ROOT, "training/data/potholes/test/images")
NEG = os.path.join(ROOT, "training/data/authenticity/test/real")


def _gpu_gb():
    """Accelerator memory (MPS unified memory is not included in process RSS)."""
    import torch
    if torch.backends.mps.is_available():
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return None


def sample(folder, n, prefix=""):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")) and f.startswith(prefix))
    step = max(1, len(files) // n)
    return [os.path.join(folder, f) for f in files[::step][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="images per class")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval", "vlm_images_report.json"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    vision.load()
    load_s = time.perf_counter() - t0
    rows = []
    # The authenticity "real" folder also holds road photos (road_*); only the
    # GenImage/ImageNet ones (gi_real_*) are non-civic negatives.
    for label, folder, prefix in (("pothole", POS, ""), ("non_civic", NEG, "gi_real_")):
        for path in sample(folder, args.n, prefix):
            with open(path, "rb") as f:
                data = f.read()
            t = time.perf_counter()
            r = vision.analyze_images([data], use_vlm=True)
            ms = (time.perf_counter() - t) * 1000
            ids = [p.subcategory_id for p in r.problems]
            rows.append({"file": os.path.relpath(path, ROOT), "label": label, "status": r.status.value,
                         "is_civic_scene": r.is_civic_scene, "problems": ids, "scene": r.scene_description,
                         "quality": r.image_quality.value, "latency_ms": round(ms), "warnings": r.warnings})
            print(f"{label:9} {r.status.value:22} civic={r.is_civic_scene!s:5} {ms/1000:5.1f}s {ids} "
                  f"| {(r.scene_description or '')[:70]}", flush=True)
    pos = [r for r in rows if r["label"] == "pothole"]
    neg = [r for r in rows if r["label"] == "non_civic"]
    lat = [r["latency_ms"] for r in rows]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    report = {
        "model": vision.metadata(),
        "max_image_side": __import__("backend.config", fromlist=["settings"]).settings.AI_VLM_MAX_IMAGE_SIDE,
        "n_pothole": len(pos), "n_non_civic": len(neg),
        "valid_json_rate": round(sum(r["status"] == StageStatus.ok.value for r in rows) / len(rows), 3),
        "pothole_recall": round(sum("ROAD.POTHOLE" in r["problems"] for r in pos) / len(pos), 3),
        "road_problem_recall": round(sum(any(p.startswith("ROAD.") for p in r["problems"]) for r in pos) / len(pos), 3),
        "pothole_scene_marked_civic": round(sum(r["is_civic_scene"] is True for r in pos) / len(pos), 3),
        "non_civic_specificity": round(sum(r["is_civic_scene"] is False and not r["problems"] for r in neg) / len(neg), 3),
        "non_civic_false_problem_rate": round(sum(bool(r["problems"]) for r in neg) / len(neg), 3),
        "model_load_s": round(load_s, 1),
        "latency_ms_median": round(statistics.median(lat)),
        "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))]),
        "peak_rss_gb": round(peak, 2),
        "gpu_driver_allocated_gb": _gpu_gb(),
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for k, v in report.items():
        if k not in ("rows", "model"):
            print(f"{k:28} {v}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
