"""
bootstrap_project.py
====================
One-shot script that prepares the entire project for execution.
Creates directories, generates sample images, runs ELA, and builds seed CSVs.
Run this ONCE before attempting any training scripts.

Usage:
    python bootstrap_project.py
"""
import os
import csv
import random
import shutil
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(".")
DATASET_DIR = BASE / "civic_dataset"
AUTH_DIR = DATASET_DIR / "authentic"
TAMP_DIR = DATASET_DIR / "tampered"
ELA_OUT = BASE / "ela_output"

CATEGORIES = ["pothole", "garbage", "broken_pipe", "streetlight"]
COLORS = {
    "pothole": [(80, 80, 80), (120, 100, 70), (60, 50, 40)],
    "garbage": [(30, 120, 30), (150, 140, 50), (200, 180, 100)],
    "broken_pipe": [(50, 80, 160), (70, 130, 180), (100, 100, 120)],
    "streetlight": [(40, 40, 60), (180, 180, 50), (80, 80, 90)],
}

def generate_synthetic_image(category, idx, size=(640, 480)):
    """Creates a synthetic civic complaint image with shapes and noise."""
    bg_color = random.choice(COLORS[category])
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    # Add random shapes simulating scene elements
    for _ in range(random.randint(8, 20)):
        x1 = random.randint(0, size[0] - 50)
        y1 = random.randint(0, size[1] - 50)
        x2 = x1 + random.randint(20, 150)
        y2 = y1 + random.randint(20, 150)
        
        color = tuple(random.randint(0, 255) for _ in range(3))
        shape = random.choice(["rect", "ellipse", "line"])
        if shape == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))
    
    # Add category label text
    try:
        draw.text((10, 10), f"{category}_{idx}", fill=(255, 255, 255))
    except Exception:
        pass

    # Add Gaussian noise
    arr = np.array(img)
    noise = np.random.normal(0, 15, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_copy_move(img):
    """Simple copy-move forgery: copies a patch from one region to another."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    patch_h, patch_w = h // 4, w // 4
    
    sy, sx = random.randint(0, h - patch_h - 1), random.randint(0, w - patch_w - 1)
    dy, dx = random.randint(0, h - patch_h - 1), random.randint(0, w - patch_w - 1)
    
    arr[dy:dy+patch_h, dx:dx+patch_w] = arr[sy:sy+patch_h, sx:sx+patch_w]
    return Image.fromarray(arr)


def apply_splicing(img1, img2):
    """Splicing forgery: pastes a region from img2 onto img1."""
    arr1 = np.array(img1)
    arr2 = np.array(img2.resize(img1.size))
    h, w = arr1.shape[:2]
    
    patch_h, patch_w = h // 3, w // 3
    sy, sx = random.randint(0, h - patch_h - 1), random.randint(0, w - patch_w - 1)
    
    arr1[sy:sy+patch_h, sx:sx+patch_w] = arr2[sy:sy+patch_h, sx:sx+patch_w]
    return Image.fromarray(arr1)


def step1_create_directories():
    """Create all required folder structures."""
    print("\n[Step 1] Creating directory structures...")
    for d in [AUTH_DIR, TAMP_DIR, ELA_OUT / "authentic", ELA_OUT / "tampered"]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")


def step2_generate_images(num_per_category=10):
    """Generate synthetic authentic and tampered images."""
    print(f"\n[Step 2] Generating {num_per_category * len(CATEGORIES)} authentic + tampered images...")
    labels = []
    
    all_authentic = []
    
    for cat in CATEGORIES:
        for i in range(num_per_category):
            img = generate_synthetic_image(cat, i)
            fname = f"{cat}_{i:03d}.jpg"
            path = AUTH_DIR / fname
            img.save(path, "JPEG", quality=95)
            labels.append([str(path), 0, "None"])
            all_authentic.append((path, img))
    
    print(f"  ✓ {len(all_authentic)} authentic images saved")
    
    # Generate tampered versions
    tampered_count = 0
    for path, img in all_authentic:
        # Copy-move forgery
        cm_img = apply_copy_move(img)
        cm_name = f"cm_{path.stem}.jpg"
        cm_path = TAMP_DIR / cm_name
        cm_img.save(cm_path, "JPEG", quality=95)
        labels.append([str(cm_path), 1, "Copy-Move"])
        tampered_count += 1
    
    # Splice some pairs
    for i in range(0, len(all_authentic) - 1, 2):
        sp_img = apply_splicing(all_authentic[i][1], all_authentic[i+1][1])
        sp_name = f"sp_{all_authentic[i][0].stem}.jpg"
        sp_path = TAMP_DIR / sp_name
        sp_img.save(sp_path, "JPEG", quality=95)
        labels.append([str(sp_path), 1, "Splicing"])
        tampered_count += 1
    
    print(f"  ✓ {tampered_count} tampered images saved")
    
    # Write labels CSV
    csv_path = DATASET_DIR / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "manipulation_type"])
        writer.writerows(labels)
    print(f"  ✓ Labels CSV: {csv_path} ({len(labels)} rows)")


def step3_run_ela():
    """Run ELA pipeline on generated dataset to populate ela_output/."""
    print("\n[Step 3] Running ELA preprocessing pipeline...")
    
    # Process authentic images -> ela_output/authentic/
    from ela_pipeline import batch_process_folder
    batch_process_folder(str(AUTH_DIR), str(ELA_OUT / "authentic"), quality=95)
    batch_process_folder(str(TAMP_DIR), str(ELA_OUT / "tampered"), quality=95)
    
    auth_count = len(list((ELA_OUT / "authentic").glob("*.jpg")))
    tamp_count = len(list((ELA_OUT / "tampered").glob("*.jpg")))
    print(f"  ✓ ELA output: {auth_count} authentic, {tamp_count} tampered")


def step4_create_complaints_csv():
    """Generate complaints_text.csv for IndicBERT training."""
    print("\n[Step 4] Generating complaints_text.csv...")
    
    complaints = {
        "Roads": [
            "There is a huge pothole on MG Road causing traffic jams.",
            "सड़क पर बहुत बड़ा गड्ढा है जो दुर्घटनाओं का कारण बन रहा है।",
            "రోడ్డు పూర్తిగా పాడైపోయింది, వాహనాలు నడవడం కష్టంగా ఉంది.",
            "रस्त्यावर मोठा खड्डा पडला आहे, वाहतूक विस्कळीत झाली आहे.",
            "রাস্তায় বড় গর্ত হয়েছে, গাড়ি চলাচল কঠিন হয়ে গেছে।",
            "റോഡിൽ വലിയ കുഴി വീണു, വാഹനങ്ങൾ ഓടിക്കാൻ കഴിയുന്നില്ല.",
            "Road surface is completely damaged near the school zone.",
            "Potholes have made the entire lane inaccessible for vehicles.",
        ],
        "Sanitation": [
            "Garbage hasn't been collected from sector 4 in three days.",
            "कूड़ा तीन दिनों से नहीं उठाया गया है, बदबू आ रही है।",
            "చెత్త సేకరణ మూడు రోజులుగా జరగలేదు.",
            "तीन दिवसांपासून कचरा उचलला नाही, दुर्गंधी येत आहे.",
            "তিন দিন ধরে আবর্জনা তোলা হয়নি, দুর্গন্ধ ছড়াচ্ছে।",
            "മാലിന്യം മൂന്ന് ദിവസമായി ശേഖരിച്ചിട്ടില്ല.",
            "Overflowing garbage bins near the park are a health hazard.",
            "Waste dump site is attracting stray animals in residential area.",
        ],
        "Water": [
            "Water supply is contaminated and smelling horrible locally.",
            "पानी की सप्लाई में गंदगी आ रही है, पीने योग्य नहीं है।",
            "నీటి సరఫరాలో మురికి నీరు వస్తోంది.",
            "पाण्याच्या पुरवठ्यात घाण पाणी येत आहे.",
            "জলের সাপ্লাইতে নোংরা জল আসছে।",
            "കുടിവെള്ള വിതരണത്തിൽ മലിനജലം വരുന്നു.",
            "Major pipe burst flooding the street with dirty water.",
            "No water supply for the past 48 hours in our locality.",
        ],
        "Electricity": [
            "Streetlights are completely dead in the entire block.",
            "बिजली गुल है, पूरा इलाका अंधेरे में है।",
            "వీధి దీపాలు పనిచేయడం లేదు, రాత్రి అంధకారంగా ఉంది.",
            "वीज गेली आहे, संपूर्ण भाग अंधारात आहे.",
            "রাস্তার আলো নষ্ট হয়ে গেছে, রাতে অন্ধকার থাকে।",
            "തെരുവ് വിളക്കുകൾ പ്രവർത്തിക്കുന്നില്ല, രാത്രിയിൽ ഇരുട്ടാണ്.",
            "Exposed electric wires dangling near the playground.",
            "Power outage lasting more than 24 hours in ward 7.",
        ],
    }
    
    rows = []
    for label_name, texts in complaints.items():
        for t in texts:
            rows.append({"text": t, "label_name": label_name})
    
    # Multiply to get decent training size
    expanded = rows * 20
    random.shuffle(expanded)
    
    import pandas as pd
    df = pd.DataFrame(expanded)
    df.to_csv("complaints_text.csv", index=False, encoding="utf-8")
    print(f"  ✓ complaints_text.csv ({len(df)} rows across 4 categories, 6 languages)")


def step5_summary():
    """Print final readiness check."""
    print("\n" + "=" * 60)
    print("PROJECT BOOTSTRAP COMPLETE")
    print("=" * 60)
    
    checks = {
        "civic_dataset/authentic/": AUTH_DIR.exists() and any(AUTH_DIR.iterdir()),
        "civic_dataset/tampered/": TAMP_DIR.exists() and any(TAMP_DIR.iterdir()),
        "ela_output/authentic/": (ELA_OUT / "authentic").exists(),
        "ela_output/tampered/": (ELA_OUT / "tampered").exists(),
        "complaints_text.csv": Path("complaints_text.csv").exists(),
    }
    
    for name, ok in checks.items():
        status = "✓ READY" if ok else "✗ MISSING"
        print(f"  [{status}] {name}")
    
    print("\nYou can now run:")
    print("  python train_forgery_cnn.py    # CNN image forgery classifier")
    print("  python train_indicbert.py      # IndicBERT text classifier")
    print("  python train_priority_rf.py    # Random Forest priority scorer")
    print("  python simulate_bengaluru.py   # End-to-end integration test")
    print("  python ml_evaluate_baselines.py # ML benchmark comparisons")


if __name__ == "__main__":
    print("=" * 60)
    print("CIVIC COMPLAINT FORENSICS — PROJECT BOOTSTRAP")
    print("=" * 60)
    
    step1_create_directories()
    step2_generate_images(num_per_category=10)
    step3_run_ela()
    step4_create_complaints_csv()
    step5_summary()
