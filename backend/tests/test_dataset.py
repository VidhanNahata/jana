"""training/civic_ai/dataset.py: validation, exclusion, de-duplication, leakage-safe split."""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("civic_dataset", os.path.join(ROOT, "training/civic_ai/dataset.py"))
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


def rec(i, text, sub="ROAD.POTHOLE", **kw):
    return {"id": str(i), "text": text, "subcategory_id": sub, "source": "citizen_submission", "reviewed": True, **kw}


def test_validation_and_exclusion():
    ok, bad = ds.validate([rec(1, "pothole"), rec(1, "dup id"), {"id": "x", "text": "", "subcategory_id": "NOPE"}])
    assert len(ok) == 1 and len(bad) == 2
    keep, excluded = ds.trainable([rec(1, "a"), rec(2, "b", reviewed=False), rec(3, "c", source="developer_authored")])
    assert [r["id"] for r in keep] == ["1"] and sum(excluded.values()) == 2


def test_dedupe_exact_near_and_conflicts():
    rs = [rec(1, "Big pothole on MG Road"), rec(2, "big  pothole on mg road"),
          rec(3, "Big pothole on MG Road!!"), rec(4, "garbage", "WASTE.UNCOLLECTED"), rec(5, "Garbage", "ROAD.POTHOLE")]
    unique, report = ds.dedupe(rs)
    assert report["exact_duplicates_removed"] == 1
    assert len(report["label_conflicts_dropped"]) == 1
    clusters = {r["id"]: r["_cluster"] for r in unique}
    assert clusters["1"] == clusters["3"]          # near-duplicates share a cluster


def test_split_keeps_groups_and_duplicates_together():
    rs = [rec(i, f"pothole number {i} in ward {i % 7}", group=f"ward{i % 7}") for i in range(200)]
    unique, _ = ds.dedupe(rs)
    parts = ds.split(unique)
    where = {}
    for name, items in parts.items():
        for r in items:
            where.setdefault(r["group"], set()).add(name)
    assert all(len(s) == 1 for s in where.values())
    assert sum(len(v) for v in parts.values()) == len(unique)
    assert ds.split(unique) == parts                # deterministic
