"""
Versioned local knowledge: taxonomy, authority registry, lexicon, gazetteer.

All files live in backend/ai/data/ (paths overridable in settings) and are
loaded once; nothing is fetched at runtime.
"""

import fnmatch
import json
import os
from functools import lru_cache

from backend.config import settings

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load(path_setting: str, default_name: str) -> dict:
    path = path_setting or os.path.join(DATA_DIR, default_name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def taxonomy() -> dict:
    return _load(settings.AI_TAXONOMY_PATH, "taxonomy.json")


@lru_cache(maxsize=1)
def registry() -> dict:
    return _load(settings.AI_REGISTRY_PATH, "authority_registry.json")


@lru_cache(maxsize=1)
def lexicon() -> dict:
    return _load(settings.AI_LEXICON_PATH, "lexicon.json")


@lru_cache(maxsize=1)
def gazetteer() -> dict:
    return _load(settings.AI_GAZETTEER_PATH, "gazetteer.json")


@lru_cache(maxsize=1)
def _subcategories() -> dict:
    out = {}
    for cat in taxonomy()["categories"]:
        for sub in cat["subcategories"]:
            out[sub["id"]] = {**sub, "category_id": cat["id"], "category_name_en": cat["name_en"],
                              "category_name_kn": cat["name_kn"]}
    return out


def subcategory(sub_id: str) -> dict | None:
    return _subcategories().get(sub_id)


def civic_subcategory_ids() -> list[str]:
    """Every in-scope subcategory (excludes OUT_OF_SCOPE.*)."""
    return [s for s in _subcategories() if not s.startswith("OUT_OF_SCOPE.")]


def category_of(sub_id: str) -> str:
    return sub_id.split(".", 1)[0]


@lru_cache(maxsize=1)
def _authorities() -> dict:
    reg = registry()
    out = {a["id"]: a for a in reg["authorities"]}
    corp = reg.get("city_corporations_outside_bengaluru", {})
    for item in corp.get("items", []):
        out[item["id"]] = {
            "id": item["id"], "official_name": item["official_name"], "name_kn": None,
            "aliases": item.get("aliases", []), "authority_type": "city_corporation", "role": "maintenance",
            "handles": ["ROAD.*", "DRAIN.*", "WATER.*", "WASTE.*", "LIGHT.STREETLIGHT_OUT", "TRANSPORT.SIGN_DAMAGE",
                        "TRANSPORT.CROSSING", "TRANSPORT.BUS_SHELTER", "TRANSPORT.BARRIER_DAMAGE", "PARK.*", "TREE.*",
                        "BUILDING.*", "STORMWATER.*", "ACCESS.*", "OTHER.CIVIC_INFRA"],
            "jurisdiction": {"level": "city_corporation", "city": item["city"], "district": item["district"],
                             "verification": corp.get("verification", "unverified")},
            "official_source_url": None, "complaint_url": None, "last_verified": None,
            "verification": {"url": "unverified", "method": corp.get("note", "")},
            "alternatives": ["KA.DMA"], "escalation": "KA.DMA",
            "routing_notes": "Name and responsibilities unverified; confirm before relying on this destination.",
        }
    return out


def authority(authority_id: str) -> dict | None:
    return _authorities().get(authority_id)


def all_authorities() -> dict:
    return _authorities()


def handles(authority_id: str, sub_id: str) -> bool:
    a = authority(authority_id)
    return bool(a) and any(fnmatch.fnmatchcase(sub_id, pattern) for pattern in a.get("handles", []))


def versions() -> dict:
    return {
        "taxonomy": taxonomy()["version"],
        "registry": registry()["version"],
        "lexicon": lexicon()["version"],
        "gazetteer": gazetteer()["version"],
    }
