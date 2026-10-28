"""
Install the pretrained CLIP image/text encoder used by the issue classifier (training/issues/train_clip_head.py).

Like scripts/install_models.py this is the ONLY step that touches the network; the backend later loads
the files strictly offline. Model: openai/clip-vit-base-patch32 (MIT licence, ~600 MB on disk).

The revision is NOT pinned here (I could not look up a commit hash): the script records the commit it
downloaded and the SHA-256 of the weights in models/CLIP_MANIFEST.json. To freeze the model, copy that
commit hash into REVISION below. `--verify` re-checks the files offline.

Usage:
    pip install huggingface_hub
    python scripts/install_clip.py
    python scripts/install_clip.py --verify
"""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "openai/clip-vit-base-patch32"
REVISION = "main"            # replace with the commit recorded in models/CLIP_MANIFEST.json to pin it
TARGET = os.path.join(ROOT, "models", "clip-vit-base-patch32")
MANIFEST = os.path.join(ROOT, "models", "CLIP_MANIFEST.json")
WEIGHTS = "model.safetensors"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    if not (os.path.exists(MANIFEST) and os.path.exists(os.path.join(TARGET, WEIGHTS))):
        print("CLIP is not installed (run: python scripts/install_clip.py)")
        return 1
    expected = json.load(open(MANIFEST))["sha256"][WEIGHTS]
    ok = sha256(os.path.join(TARGET, WEIGHTS)) == expected
    print("CLIP weights", "OK" if ok else "CHECKSUM MISMATCH")
    return 0 if ok else 1


def install():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub  first")
    commit = HfApi().model_info(REPO, revision=REVISION).sha
    snapshot_download(REPO, revision=commit, local_dir=TARGET,
                      allow_patterns=["*.json", "*.txt", WEIGHTS])
    if not os.path.exists(os.path.join(TARGET, WEIGHTS)):
        sys.exit(f"{REPO}@{commit} has no {WEIGHTS}; check the repository files")
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump({"repo": REPO, "commit": commit, "license": "MIT",
               "sha256": {WEIGHTS: sha256(os.path.join(TARGET, WEIGHTS))}}, open(MANIFEST, "w"), indent=2)
    print(f"installed {REPO}@{commit[:10]} -> {TARGET}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    sys.exit(verify() if ap.parse_args().verify else install())
