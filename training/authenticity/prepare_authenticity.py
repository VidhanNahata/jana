"""
Build the real / ai_generated / screen_photo dataset for the authenticity classifier.

Sources (training/data/raw/authenticity/):
  - tiny_genimage/  GenImage subset (CC BY-NC-SA 4.0): ImageNet reals + ADM, GLIDE,
    Midjourney v5, SD 1.5, VQDM, Wukong fakes. BigGAN (128 px) is skipped.
  - openfake_tiny/  OpenFake (CC BY-NC 4.0): Pexels / LAION reals + ~100 modern
    generators (FLUX, SD3.5, SDXL family, Midjourney 6/7, DALL-E 3, GPT-image,
    Imagen, Ideogram, ...). Anime / NSFW-named models are skipped.
  - local_sd/       AI pothole/road images generated locally (optional, see generate_ai_roads.py)
  - training/data/potholes/  real road photos (pothole dataset splits)
screen_photo images are simulated from real and AI images (recapture.py).

Held-out test sets measure generalisation, not memorisation:
  test/                 same generators / sources as training
  test_unseen_wukong/   GenImage Wukong (never seen in training) vs ImageNet reals
  test_openfake/        OpenFake test split: mostly newer generators than train
  test_real_screens/    (optional) real phone photos of screens, put in
                        training/data/raw/authenticity/real_screens/

All images are stored as JPEG q97 with the longer side capped, identically for
every class; training adds its own random rescale + JPEG on top.
"""

import glob
import io
import os
import random
import re
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recapture import simulate_recapture  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "..", "data", "raw", "authenticity")
POTHOLES = os.path.join(ROOT, "..", "data", "potholes")
OUT = os.path.join(ROOT, "..", "data", "authenticity")
SEED = 7
MAX_SIDE = 1024
GENIMAGE_NAMES = {0: "real", 1: "adm", 2: "biggan", 3: "glide", 4: "midjourney", 5: "sd14", 6: "sd15", 7: "vqdm", 8: "wukong"}
SKIP_MODEL = re.compile(r"nsfw|anime|animagine|anima-|illustrious|pony|ghibli|noobai|hentai|waifu|tiny-random", re.I)


def load_image(data):
    img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        # Very large photos (Pexels 6-8k) get a random cap so resolution isn't a class cue
        cap = random.randint(768, MAX_SIDE)
        img.thumbnail((cap, cap), Image.LANCZOS)
    return img


def genimage(shards, generators):
    """Yield (PIL image, generator name) from GenImage parquet shards."""
    for f in shards:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=["image", "generator"]):
            for row in batch.to_pylist():
                if row["generator"] in generators:
                    yield row["image"]["bytes"], GENIMAGE_NAMES[row["generator"]]


def openfake(files):
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=16, columns=["image", "label", "model"]):
            for row in batch.to_pylist():
                if row["label"] == "fake" and SKIP_MODEL.search(row["model"] or ""):
                    continue
                yield row["image"]["bytes"], row["label"], row["model"]


def road_photos(split, n):
    files = sorted(glob.glob(os.path.join(POTHOLES, split, "images", "*.jpg")))
    random.shuffle(files)
    return files[:n]


class Writer:
    def __init__(self):
        self.counts = {}

    def save(self, img, split, cls, name):
        d = os.path.join(OUT, split, cls)
        os.makedirs(d, exist_ok=True)
        img.save(os.path.join(d, name + ".jpg"), quality=97)
        self.counts[(split, cls)] = self.counts.get((split, cls), 0) + 1

    def recapture(self, img, split, name):
        sim = simulate_recapture(np.asarray(img))
        self.save(Image.fromarray(sim), split, "screen_photo", name)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    w = Writer()
    gi = sorted(glob.glob(os.path.join(RAW, "tiny_genimage", "*.parquet")))
    seen_gens = {1, 3, 4, 6, 7}  # ADM, GLIDE, Midjourney, SD1.5, VQDM

    def split_of(i, val_every=10):
        return "val" if i % val_every == 0 else "train"

    # Screen photos are simulated from a fraction of every source, so the
    # model sees the same kinds of content both directly and on a screen.
    def emit(img, split, cls, name, recapture_p):
        if random.random() < recapture_p:
            w.recapture(img, split, name + "_scr")
        else:
            w.save(img, split, cls, name)

    # 1. GenImage, shards 0-4 -> train/val, shard 5 -> test (+ unseen Wukong)
    for i, (data, gen) in enumerate(genimage(gi[:5], {0} | seen_gens)):
        if gen == "real" and random.random() > 0.5:  # 5000 ImageNet reals is more than needed
            continue
        cls = "real" if gen == "real" else "ai_generated"
        emit(load_image(data), split_of(i), cls, f"gi_{gen}_{i:05d}", 0.25)
    for i, (data, gen) in enumerate(genimage(gi[5:], {0, 8} | seen_gens)):
        img = load_image(data)
        if gen == "wukong":
            w.save(img, "test_unseen_wukong", "ai_generated", f"gi_{gen}_{i:05d}")
        elif gen == "real":
            if i % 4 == 0:
                w.save(img, "test_unseen_wukong", "real", f"gi_{gen}_{i:05d}")
            elif i % 4 == 1:
                emit(img, "test", "real", f"gi_{gen}_{i:05d}", 0.3)
        elif i % 2 == 0:
            emit(img, "test", "ai_generated", f"gi_{gen}_{i:05d}", 0.3)

    # 2. OpenFake: train + validation splits -> train/val; test split -> test_openfake
    of = sorted(glob.glob(os.path.join(RAW, "openfake_tiny", "*.parquet")))
    trainval = [f for f in of if "/test-" not in f]
    for i, (data, lab, model) in enumerate(openfake(trainval)):
        cls = "real" if lab == "real" else "ai_generated"
        img = load_image(data)
        split = split_of(i)
        emit(img, split, cls, f"of_{i:05d}", 0.2)
        if cls == "ai_generated" and split == "train":
            # Modern generators matter most and are scarce: add a second copy
            # (training applies different random crops / compression to it)
            w.save(img, split, cls, f"of_{i:05d}_b")
    for i, (data, lab, model) in enumerate(openfake([f for f in of if "/test-" in f])):
        w.save(load_image(data), "test_openfake", "real" if lab == "real" else "ai_generated", f"of_{i:05d}")

    # 3. Real road photos (same split as the pothole dataset)
    for split, n, out_split in (("train", 3500, "train"), ("val", 400, "val"), ("test", 500, "test")):
        for i, f in enumerate(road_photos(split, n)):
            emit(load_image(f), out_split, "real", f"road_{os.path.splitext(os.path.basename(f))[0]}", 0.3)

    # 4. Locally generated AI road / pothole images (optional)
    local = sorted(glob.glob(os.path.join(RAW, "local_sd", "*", "*.png")))
    for i, f in enumerate(local):
        # Hold out one whole local generator folder for test if there are several
        gen = os.path.basename(os.path.dirname(f))
        split = "test" if gen.endswith("_test") else split_of(i)
        img = load_image(f)
        if np.asarray(img).std() < 5:  # fp16 on MPS can occasionally output black frames
            continue
        emit(img, split, "ai_generated", f"sd_{gen}_{i:05d}", 0.25)

    # 5. Real phone photos of screens (optional; test only)
    for i, f in enumerate(sorted(glob.glob(os.path.join(RAW, "real_screens", "*")))):
        try:
            w.save(load_image(f), "test_real_screens", "screen_photo", f"rs_{i:04d}")
        except Exception:
            pass

    # Every split must have all three class folders for the dataloader
    for split in {s for s, _ in w.counts}:
        for cls in ("real", "ai_generated", "screen_photo"):
            os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)
    for k in sorted(w.counts):
        print(k, w.counts[k])


if __name__ == "__main__":
    main()
