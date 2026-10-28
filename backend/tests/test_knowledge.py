"""Integrity of the versioned taxonomy, authority registry, lexicon and gazetteer."""

import re

from backend.ai import knowledge

REQUIRED_CATEGORIES = {"ROAD", "DRAIN", "WATER", "WASTE", "LIGHT", "TRANSPORT", "PARK", "TREE", "BUILDING",
                       "STORMWATER", "ACCESS", "OTHER", "OUT_OF_SCOPE"}
LEGACY = {"pothole", "road_damage", "drainage", "sewage", "water_leak", "garbage", "waste", "streetlight",
          "electricity", "traffic_signal", "tree_fall", "other", "out_of_scope"}


def test_taxonomy_ids_are_stable_and_unique():
    tax = knowledge.taxonomy()
    assert {c["id"] for c in tax["categories"]} == REQUIRED_CATEGORIES
    ids = [s["id"] for c in tax["categories"] for s in c["subcategories"]]
    assert len(ids) == len(set(ids))
    for c in tax["categories"]:
        for s in c["subcategories"]:
            assert s["id"].startswith(c["id"] + "."), s["id"]
            assert re.fullmatch(r"[A-Z_]+\.[A-Z_]+", s["id"])
            assert s["name_en"] and s["name_kn"] and s["legacy_code"] in LEGACY and 1 <= s["severity"] <= 3


def test_every_civic_subcategory_has_lexicon_terms():
    lx = knowledge.lexicon()["subcategories"]
    for sid in knowledge.civic_subcategory_ids():
        if sid != "OTHER.CIVIC_INFRA":          # fallback, never matched directly
            assert sid in lx, f"{sid} has no lexicon entry"
    for sid in lx:
        assert knowledge.subcategory(sid), f"lexicon entry {sid} not in taxonomy"


def test_registry_entries_are_well_formed():
    ids = set()
    for aid, a in knowledge.all_authorities().items():
        assert aid == a["id"] and aid not in ids
        ids.add(aid)
        assert a["official_name"]
        v = a.get("verification") or {}
        # A URL is only offered to citizens when it was actually checked
        if a.get("official_source_url"):
            assert v.get("url") in ("url_verified", "content_verified", "unreachable", "unverified"), aid
        for pattern in a.get("handles", []):
            assert any(knowledge.handles(aid, s) for s in knowledge.civic_subcategory_ids()), (aid, pattern)
    for aid, a in knowledge.all_authorities().items():
        for alt in a.get("alternatives", []) or []:
            assert (alt if isinstance(alt, str) else alt.get("id")) in ids, (aid, alt)


def test_bbmp_is_not_a_destination():
    assert "KA.BBMP" not in knowledge.all_authorities()
    gba = knowledge.authority("KA.BLR.GBA")
    assert "BBMP" in " ".join(gba.get("aliases", []))


def test_every_civic_subcategory_has_some_authority():
    for sid in knowledge.civic_subcategory_ids():
        assert any(knowledge.handles(aid, sid) for aid in knowledge.all_authorities()), sid
