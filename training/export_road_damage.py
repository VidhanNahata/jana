"""
Export the pretrained RDD2022 road-damage detector (dronefreak/rdd2022-yolov8n,
AGPL-3.0, pinned revision in scripts/install_models.py) to ONNX and install it:
  frontend/public/models/road_damage.{onnx,json}  (dashcam mode, on-device)
  backend/ml_models/road_damage.{onnx,json}       (server re-check + priority)

Per-class thresholds and test metrics come from eval_road_damage.py's report.
Usage: python training/export_road_damage.py
"""
import json, os, shutil
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
WEIGHTS = os.path.join(PROJECT, "models", "rdd2022-yolov8n", "best.pt")
REPORT = os.path.join(ROOT, "runs", "road_damage_pretrained_eval.json")
# Model class -> app class (taxonomy subcategory and display label)
CLASSES = {
    "longitudinal_crack": ("Road crack (along the road)", "ROAD.SURFACE_DAMAGE"),
    "transverse_crack": ("Road crack (across the road)", "ROAD.SURFACE_DAMAGE"),
    "alligator_crack": ("Alligator cracking", "ROAD.SURFACE_DAMAGE"),
    "pothole": ("Pothole", "ROAD.POTHOLE"),
}


def main():
    model = YOLO(WEIGHTS)
    report = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    onnx = model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    classes = []
    for i in range(len(model.names)):
        name = model.names[i]
        label, sid = CLASSES[name]
        classes.append({"name": name, "label_en": label, "subcategory_id": sid,
                        "threshold": report.get("thresholds", {}).get(name, 0.3),
                        # Measured F1 0.009 on Indian potholes; pothole.onnx handles potholes
                        "enabled": name != "pothole"})
    config = {"inputSize": 640, "classes": classes,
              "model": "dronefreak/rdd2022-yolov8n (pretrained on RDD2022, AGPL-3.0)",
              "testMetrics": report.get("per_class")}
    for d in (os.path.join(PROJECT, "frontend", "public", "models"), os.path.join(PROJECT, "backend", "ml_models")):
        shutil.copy(onnx, os.path.join(d, "road_damage.onnx"))
        json.dump(config, open(os.path.join(d, "road_damage.json"), "w"), indent=2)
    print("exported", onnx, os.path.getsize(onnx) / 1e6, "MB")


if __name__ == "__main__":
    main()
