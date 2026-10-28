"""
Re-check the official URLs in the authority registry (network required; run by
a maintainer, never by the backend).

    python scripts/verify_registry.py [--out registry_check.json]

For every authority with an official_source_url it records the HTTP status,
final URL after redirects and page title, and flags entries whose recorded
verification no longer matches (e.g. marked url_verified but now unreachable).
It does NOT edit backend/ai/data/authority_registry.json: a person reviews
the report, updates the entry (verification, last_verified) and bumps the
registry version.
"""

import argparse
import json
import os
import re
import ssl
import sys
import urllib.request
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.ai import knowledge  # noqa: E402


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def check(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (registry verification)"})
    try:
        with urllib.request.urlopen(req, context=_ctx(), timeout=20) as r:
            body = r.read(200_000).decode("utf-8", "replace")
            title = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
            return {"http_status": r.status, "final_url": r.geturl(),
                    "title": " ".join(title.group(1).split())[:150] if title else None}
    except Exception as exc:
        return {"http_status": None, "error": f"{type(exc).__name__}: {exc}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="registry_check.json")
    args = ap.parse_args()
    rows = []
    for aid, a in knowledge.all_authorities().items():
        url = a.get("official_source_url")
        recorded = (a.get("verification") or {}).get("url")
        if not url:
            rows.append({"id": aid, "url": None, "recorded": recorded, "result": "no url"})
            continue
        res = check(url)
        reachable = res.get("http_status") == 200
        mismatch = (recorded in ("url_verified", "content_verified")) != reachable
        rows.append({"id": aid, "url": url, "recorded": recorded, **res, "needs_review": mismatch})
        print(f"{'!!' if mismatch else 'ok'} {aid:24} {res.get('http_status') or res.get('error', '')[:40]}  "
              f"{res.get('title') or ''}", flush=True)
    report = {"checked": str(date.today()), "registry_version": knowledge.versions()["registry"], "results": rows}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"{sum(r.get('needs_review', False) for r in rows)} entries need review; wrote {args.out}")


if __name__ == "__main__":
    main()
