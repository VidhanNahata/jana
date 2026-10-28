"""
Install / update the local models used by the civic AI pipeline.

This is the ONLY step that touches the network. The backend itself runs in
strict offline mode (AI_STRICT_OFFLINE=true) and never downloads models
during complaint processing.

Pinned model (edit MODELS + run again to update; old files are replaced only
after the new download verifies):
  Qwen/Qwen3-VL-2B-Instruct  (Apache-2.0)  revision 89644892e4d85e24eaac8bacfd4f463576704203

Usage:
    python scripts/install_models.py             # install / verify
    python scripts/install_models.py --verify    # checksum check only, no network
Writes models/MODEL_MANIFEST.json with revision, sizes and SHA-256 hashes.
"""

import argparse
import hashlib
import json
import os
import shutil
import ssl
import sys
import urllib.request
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MANIFEST = os.path.join(MODELS_DIR, "MODEL_MANIFEST.json")

MODELS = [
    {
        "repo": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "license": "Apache-2.0",
        "dir": "qwen3-vl-2b-instruct",
        "files": {
            # file: sha256 (None = small text file, hash recorded on install)
            "model.safetensors": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
            "config.json": None, "generation_config.json": None, "chat_template.json": None,
            "preprocessor_config.json": None, "video_preprocessor_config.json": None,
            "tokenizer.json": None, "tokenizer_config.json": None, "vocab.json": None, "merges.txt": None,
        },
    },
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, dest: str):
    """Resumable HTTP download to dest (via dest + '.part')."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r, open(part, "ab" if have else "wb") as f:
        if have and r.status != 206:  # server ignored the range: start over
            f.seek(0)
            f.truncate()
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(part, dest)


def _pinned(model: dict) -> dict:
    """Hashes recorded at install time for files without a hash in MODELS."""
    if not os.path.exists(MANIFEST):
        return {}
    with open(MANIFEST) as f:
        for m in json.load(f)["models"]:
            if m["repo"] == model["repo"] and m["revision"] == model["revision"]:
                return m["files"]
    return {}


def install(model: dict, verify_only: bool) -> dict:
    pinned = _pinned(model)
    target = os.path.join(MODELS_DIR, model["dir"])
    os.makedirs(target, exist_ok=True)
    record = {"repo": model["repo"], "revision": model["revision"], "license": model["license"],
              "path": os.path.relpath(target, ROOT), "files": {}}
    for name, expected in model["files"].items():
        path = os.path.join(target, name)
        if not os.path.exists(path):
            if verify_only:
                raise SystemExit(f"missing {path}")
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
            print(f"downloading {name} ...", flush=True)
            download(url, path)
        digest = sha256(path)
        expected = expected or pinned.get(name, {}).get("sha256")
        if expected and digest != expected:
            raise SystemExit(f"checksum mismatch for {path}: {digest} != {expected} (delete it and re-run)")
        record["files"][name] = {"sha256": digest, "bytes": os.path.getsize(path)}
        print(f"ok  {name}  {digest[:12]}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="verify installed files only (no network)")
    args = parser.parse_args()
    records = [install(m, args.verify) for m in MODELS]
    manifest = {"installed": str(date.today()), "models": records}
    if not args.verify:
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {MANIFEST}")
    print("all models verified")


if __name__ == "__main__":
    sys.exit(main())
