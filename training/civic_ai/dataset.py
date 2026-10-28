"""
Civic complaint text dataset: schema, validation, de-duplication and a
leakage-safe train/val/test split.

    python training/civic_ai/dataset.py data/civic_text/labels.jsonl --out data/civic_text/v1

Input: JSONL, one record per complaint (schema below). Only records whose
label was reviewed by a person (reviewed = true) and that are not
developer-authored can be used for training; everything else is reported and
excluded. Output: train/val/test JSONL plus a manifest with counts, label
distribution, the taxonomy version and a SHA-256 of every file.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backend.ai import knowledge  # noqa: E402
from backend.ai.text_analysis import normalize  # noqa: E402
from backend.services.ai_service import compute_similarity, get_text_embedding  # noqa: E402

SCHEMA = {
    "id": "unique record id (string)",
    "text": "complaint text as written by the citizen (string, non-empty)",
    "subcategory_id": "taxonomy subcategory ID, incl. OUT_OF_SCOPE.* (string)",
    "source": "citizen_submission | public_dataset | developer_authored",
    "reviewed": "true only if a person checked the label (bool)",
    "reviewer": "who reviewed it (string, optional)",
    "group": "leakage group, e.g. ward/locality or master ticket id (string, optional)",
    "language": "en | kn | kn_en_mixed | kn_transliterated (optional)",
    "licence": "licence of the text if from a public dataset (optional)",
}
SOURCES = {"citizen_submission", "public_dataset", "developer_authored"}
NEAR_DUPLICATE = 0.9
SPLITS = (("train", 0.7), ("val", 0.15), ("test", 0.15))


def validate(records):
    ok, problems, ids = [], [], set()
    for i, r in enumerate(records):
        err = []
        if not isinstance(r.get("id"), str) or not r["id"]:
            err.append("missing id")
        elif r["id"] in ids:
            err.append("duplicate id")
        if not isinstance(r.get("text"), str) or not r["text"].strip():
            err.append("empty text")
        if not knowledge.subcategory(r.get("subcategory_id", "")):
            err.append(f"unknown subcategory {r.get('subcategory_id')!r}")
        if r.get("source") not in SOURCES:
            err.append(f"source must be one of {sorted(SOURCES)}")
        if not isinstance(r.get("reviewed"), bool):
            err.append("reviewed must be true/false")
        if err:
            problems.append({"line": i + 1, "id": r.get("id"), "errors": err})
        else:
            ids.add(r["id"])
            ok.append(r)
    return ok, problems


def trainable(records):
    keep, excluded = [], Counter()
    for r in records:
        if not r["reviewed"]:
            excluded["not reviewed"] += 1
        elif r["source"] == "developer_authored":
            excluded["developer-authored (evaluation only)"] += 1
        else:
            keep.append(r)
    return keep, dict(excluded)


def dedupe(records):
    """Drop exact duplicates (same normalised text and label); cluster near-duplicates.

    Near-duplicates (char-3gram cosine >= NEAR_DUPLICATE) are kept but share a
    cluster id so the split puts them on the same side. Conflicting labels for
    the same text are reported and both records dropped."""
    by_text = defaultdict(list)
    for r in records:
        by_text[" ".join(normalize(r["text"]).split())].append(r)
    unique, conflicts, exact = [], [], 0
    for text, rs in by_text.items():
        labels = {r["subcategory_id"] for r in rs}
        if len(labels) > 1:
            conflicts.append({"text": rs[0]["text"], "labels": sorted(labels), "ids": [r["id"] for r in rs]})
            continue
        exact += len(rs) - 1
        unique.append(rs[0])
    vecs = [get_text_embedding(r["text"]) for r in unique]
    cluster = list(range(len(unique)))

    def find(i):
        while cluster[i] != i:
            cluster[i] = cluster[cluster[i]]
            i = cluster[i]
        return i
    near = 0
    for i in range(len(unique)):          # O(n^2): fine up to tens of thousands of records
        for j in range(i + 1, len(unique)):
            if compute_similarity(vecs[i], vecs[j]) >= NEAR_DUPLICATE:
                a, b = find(i), find(j)
                if a != b:
                    cluster[b] = a
                    near += 1
    for i, r in enumerate(unique):
        r["_cluster"] = f"dup{find(i)}"
    return unique, {"exact_duplicates_removed": exact, "near_duplicate_links": near,
                    "label_conflicts_dropped": conflicts}


def split(records, seed="civic-v1"):
    """Group-aware deterministic split: every record of a leakage group (and its
    near-duplicate cluster) lands in the same split."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in records:
        keys = [r["_cluster"]] + ([f"g:{r['group']}"] if r.get("group") else [])
        for k in keys[1:]:
            parent[find(k)] = find(keys[0])
    out = {name: [] for name, _ in SPLITS}
    for r in records:
        h = int(hashlib.sha256(f"{seed}:{find(r['_cluster'])}".encode()).hexdigest(), 16) / 2 ** 256
        acc = 0.0
        for name, frac in SPLITS:
            acc += frac
            if h < acc:
                out[name].append(r)
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", action="store_true", help="print the record schema and exit")
    args = ap.parse_args()
    with open(args.labels) as f:
        records = [json.loads(line) for line in f if line.strip()]
    valid, problems = validate(records)
    usable, excluded = trainable(valid)
    unique, dup_report = dedupe(usable)
    parts = split(unique)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"created": str(date.today()), "source_file": os.path.abspath(args.labels),
                "taxonomy_version": knowledge.versions()["taxonomy"], "schema": SCHEMA,
                "input_records": len(records), "invalid": problems, "excluded": excluded, **dup_report,
                "splits": {}}
    for name, rs in parts.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
        manifest["splits"][name] = {"n": len(rs), "labels": dict(Counter(r["subcategory_id"] for r in rs)),
                                    "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: manifest[k] for k in ("input_records", "excluded", "exact_duplicates_removed",
                                                "near_duplicate_links")}, indent=2))
    print({k: v["n"] for k, v in manifest["splits"].items()}, f"invalid={len(problems)}")


if __name__ == "__main__":
    if "--schema" in sys.argv:
        print(json.dumps(SCHEMA, indent=2))
    else:
        main()
