"""
Train the lightweight text classifier (TF-IDF character n-grams + logistic
regression) on a split produced by dataset.py.

    python training/civic_ai/train_text_classifier.py data/civic_text/v1 --out models/civic_text_clf/v1

Refuses to train unless there are enough reviewed, non-developer labels:
at least MIN_PER_CLASS training examples in each of at least MIN_CLASSES
subcategories, and a non-empty test split. A model trained on fewer examples
would report accuracy that means nothing. Writes the model, its metrics on the
held-out test split, and a manifest (data hashes, taxonomy version, sklearn
version) next to it.

LoRA fine-tuning of the VLM is deliberately not offered: it is only worth it
once (a) thousands of reviewed image+text complaints exist and (b) this
classifier and the zero-shot VLM have been measured on them and fall short.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MIN_PER_CLASS = 30
MIN_CLASSES = 5


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train, val, test = (load(os.path.join(args.split_dir, f"{s}.jsonl")) for s in ("train", "val", "test"))
    counts = Counter(r["subcategory_id"] for r in train)
    enough = {k: v for k, v in counts.items() if v >= MIN_PER_CLASS}
    if len(enough) < MIN_CLASSES or not test:
        raise SystemExit(
            f"Refusing to train: need >= {MIN_PER_CLASS} reviewed training examples in >= {MIN_CLASSES} "
            f"subcategories and a test split; have {len(enough)} such classes ({dict(counts)}), "
            f"test={len(test)}. Collect and review more labelled complaints first.")
    # Classes below the minimum are left out (the lexicon/VLM still cover them)
    keep = set(enough)
    train = [r for r in train if r["subcategory_id"] in keep]
    val = [r for r in val if r["subcategory_id"] in keep]
    test = [r for r in test if r["subcategory_id"] in keep]

    import joblib
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.pipeline import make_pipeline

    best = None
    for c in (0.3, 1.0, 3.0, 10.0):       # C chosen on the validation split only
        model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                              LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
        model.fit([r["text"] for r in train], [r["subcategory_id"] for r in train])
        f1 = f1_score([r["subcategory_id"] for r in val], model.predict([r["text"] for r in val]),
                      average="macro") if val else 0.0
        if best is None or f1 > best[0]:
            best = (f1, c, model)
    val_f1, c, model = best
    y_true = [r["subcategory_id"] for r in test]
    y_pred = model.predict([r["text"] for r in test])
    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, os.path.join(args.out, "model.joblib"))
    split_manifest = json.load(open(os.path.join(args.split_dir, "manifest.json")))
    manifest = {
        "trained": str(date.today()), "model": "tfidf-char(2-5)+logreg", "C": c, "classes": sorted(keep),
        "sklearn": sklearn.__version__, "taxonomy_version": split_manifest["taxonomy_version"],
        "data": {k: v["sha256"] for k, v in split_manifest["splits"].items()},
        "val_macro_f1": round(val_f1, 4),
        "test_macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "test_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "model_sha256": hashlib.sha256(open(os.path.join(args.out, "model.joblib"), "rb").read()).hexdigest(),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"test macro-F1 {manifest['test_macro_f1']} on {len(test)} held-out examples; wrote {args.out}")


if __name__ == "__main__":
    main()
