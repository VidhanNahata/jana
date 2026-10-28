"""
Evaluate the civic AI pipeline on a labelled JSONL set (text stages, no VLM)
and write a metrics report.

    python training/civic_ai/evaluate.py training/civic_ai/eval/dev_eval_v1.jsonl [--llm]

--llm also runs the local Qwen3-VL model on the text (translation + zero-shot
reading); without it only the lexicon stages run.

Reports macro-F1 and per-class precision/recall over subcategories, category
accuracy, out-of-scope precision/recall, routing accuracy (expected authority
or routing status), language-detection accuracy and per-case latency. The
bundled set is developer-authored (see eval/README.md): numbers on it show the
system works as designed, not real-world accuracy.
"""

import json
import os
import statistics
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
USE_LLM = "--llm" in sys.argv
if USE_LLM:
    os.environ.setdefault("AI_LLM_TEXT_CLASSIFY", "true")
os.environ.setdefault("AI_VLM_ENABLED", "true" if USE_LLM else "false")

from backend.ai import knowledge, pipeline  # noqa: E402
from backend.ai.routing_engine import CitizenLocation  # noqa: E402


def load(path):
    with open(path) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    for i, c in enumerate(cases):
        assert knowledge.subcategory(c["subcategory"]), f"case {i}: unknown subcategory {c['subcategory']}"
        if c.get("authority"):
            assert knowledge.authority(c["authority"]), f"case {i}: unknown authority {c['authority']}"
    return cases


def prf(gold, pred, label):
    tp = sum(g == label and p == label for g, p in zip(gold, pred))
    fp = sum(g != label and p == label for g, p in zip(gold, pred))
    fn = sum(g == label and p != label for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp + fn


def main(path, use_llm=False):
    cases = load(path)
    gold, pred, rows, latencies = [], [], [], []
    route_ok = route_n = lang_ok = lang_n = 0
    for c in cases:
        t0 = time.perf_counter()
        r = pipeline.analyze([], c["text"], CitizenLocation(**c["location"]), run_vlm=use_llm, run_detectors=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        got = r.classification.subcategory_id
        gold.append(c["subcategory"])
        pred.append(got)
        row = {"text": c["text"], "expected": c["subcategory"], "got": got, "ok": got == c["subcategory"]}
        if c.get("authority") or c.get("routing_status"):
            route_n += 1
            ok = True
            if c.get("authority"):
                ok &= r.routing.authority_id == c["authority"]
            if c.get("routing_status"):
                ok &= r.routing.status.value == c["routing_status"]
            route_ok += ok
            row.update(expected_authority=c.get("authority"), got_authority=r.routing.authority_id,
                       routing_status=r.routing.status.value, routing_ok=ok)
        if c.get("language"):
            lang_n += 1
            lang_ok += r.text_analysis.language.value == c["language"]
            row.update(expected_language=c["language"], got_language=r.text_analysis.language.value)
        rows.append(row)

    labels = sorted(set(gold))
    per_class = {lab: dict(zip(("precision", "recall", "f1", "support"), prf(gold, pred, lab))) for lab in labels}
    macro_f1 = statistics.mean(v["f1"] for v in per_class.values())
    cat = lambda s: s.split(".")[0]
    oos_gold = [g.startswith("OUT_OF_SCOPE") for g in gold]
    oos_pred = [p.startswith("OUT_OF_SCOPE") for p in pred]
    oos_p, oos_r, _, _ = prf(oos_gold, oos_pred, True)
    report = {
        "dataset": os.path.relpath(path, ROOT),
        "dataset_note": "developer-authored; not real citizen complaints",
        "n": len(cases),
        "knowledge_versions": knowledge.versions(),
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "mode": "text: lexicon + local LLM reading" if use_llm else "text: lexicon only (VLM and detectors off)",
        "subcategory_accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 3),
        "subcategory_macro_f1": round(macro_f1, 3),
        "category_accuracy": round(sum(cat(g) == cat(p) for g, p in zip(gold, pred)) / len(gold), 3),
        "out_of_scope_precision": round(oos_p, 3),
        "out_of_scope_recall": round(oos_r, 3),
        "routing_accuracy": round(route_ok / route_n, 3) if route_n else None,
        "routing_cases": route_n,
        "language_accuracy": round(lang_ok / lang_n, 3) if lang_n else None,
        "latency_ms_median": round(statistics.median(latencies), 2),
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2),
        "confusions": Counter(f"{g} -> {p}" for g, p in zip(gold, pred) if g != p).most_common(),
        "per_class": {k: {m: round(v, 3) if isinstance(v, float) else v for m, v in d.items()}
                      for k, d in per_class.items()},
        "cases": rows,
    }
    out = os.path.splitext(path)[0] + ("_report_llm.json" if use_llm else "_report.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for k in ("n", "subcategory_accuracy", "subcategory_macro_f1", "category_accuracy", "out_of_scope_precision",
              "out_of_scope_recall", "routing_accuracy", "routing_cases", "language_accuracy",
              "latency_ms_median", "latency_ms_p95"):
        print(f"{k:24} {report[k]}")
    print("misclassified:")
    for r in rows:
        if not r["ok"]:
            print(f"  {r['expected']:28} -> {r['got']:28} | {r['text']}")
    print("routing errors:")
    for r in rows:
        if r.get("routing_ok") is False:
            print(f"  {r['expected_authority']} / got {r['got_authority']} ({r['routing_status']}) | {r['text']}")
    print("language errors:")
    for r in rows:
        if r.get("expected_language") and r["expected_language"] != r["got_language"]:
            print(f"  {r['expected_language']} / got {r['got_language']} | {r['text']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else os.path.join(os.path.dirname(__file__), "eval", "dev_eval_v2_heldout.jsonl"), USE_LLM)
