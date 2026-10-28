"""
Stage 2: text understanding and language normalisation.

Deterministic and model-free, so it always works offline: language/script
detection (Kannada, English, code-mixed, transliterated Kannada), Unicode
normalisation, multilingual lexicon matching against the taxonomy, and
extraction of location mentions, duration and severity cues.
The pipeline may add a local-LLM English translation for Kannada text
(vision.translate_to_english); matching then runs on both texts.
"""

import math
import re
import unicodedata

from backend.ai import knowledge
from backend.ai.schema import Language, LocationMentions, StageStatus, TermMatch, TextAnalysis, TextCandidate

KANNADA = re.compile(r"[ಀ-೿]")
LATIN_WORD = re.compile(r"[a-z]{2,}")
OTHER_INDIC = re.compile(r"[ऀ-౿ഀ-෿]")  # Devanagari..Telugu, Malayalam, Sinhala
JOINERS = re.compile(r"[​‌‍﻿]")
EN_STOPWORDS = {"the", "is", "a", "an", "and", "of", "in", "on", "to", "for", "near", "this", "there", "it",
                "has", "have", "been", "not", "with", "at", "from", "our", "my", "very", "since", "are", "was"}
PIN = re.compile(r"\b(5[6-9]\d)\s?(\d{3})\b")
DURATION_PATTERNS = [
    re.compile(r"\b(?:for|since|past|last)?\s*((?:\d+|one|two|three|four|five|six|a few|few|several|many)\s+"
               r"(?:day|week|month|year)s?)\b"),
    re.compile(r"\bsince\s+(yesterday|last\s+(?:week|month|year)|\d{1,2}(?:st|nd|rd|th)?\s+\w+)\b"),
    re.compile(r"(\d+\s*(?:ದಿನ|ವಾರ|ತಿಂಗಳು|ತಿಂಗಳಿಂದ|ವರ್ಷ)\S*)"),
    re.compile(r"\b(\d+\s*(?:dina|dinadinda|vaara|tingalu|thingalu|varsha)\w*)\b"),
]
LANDMARK = re.compile(r"\b(?:near|opposite|opp\.?|in front of|next to|behind|beside)\s+((?:the\s+)?[\w'.-]+(?:\s+[\w'.-]+){0,3})")
LANDMARK_TL = re.compile(r"((?:[\w'.-]+\s+)?[\w'.-]+)\s+(?:hattira|bali|munde|pakka|edurugade)\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = JOINERS.sub("", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> Language:
    if not text.strip():
        return Language.unknown
    kn = len(KANNADA.findall(text))
    latin_words = LATIN_WORD.findall(text)
    if OTHER_INDIC.search(text) and not kn:
        return Language.other
    if kn and len(latin_words) >= 2:
        return Language.kn_en_mixed
    if kn:
        return Language.kn
    if not latin_words:
        return Language.unknown
    glossary = knowledge.lexicon()["transliteration_glossary"]
    tl_hits = sum(1 for w in latin_words if w in glossary)
    stop_ratio = sum(1 for w in latin_words if w in EN_STOPWORDS) / len(latin_words)
    if tl_hits >= 2 or (tl_hits >= 1 and stop_ratio < 0.2):
        return Language.kn_transliterated
    return Language.en


def _term_regex(term: str, script: str) -> re.Pattern:
    term = normalize(term)
    # Allow up to two words between the words of a phrase ("rasthe tumba haalagide")
    body = r"(?:\s+\S+){0,2}?\s+".join(re.escape(part) for part in term.split())
    if script == "kn":
        # Kannada inflects with suffixes: match at a word start, any ending
        return re.compile(r"(?:^|[\s\W])" + body)
    return re.compile(r"\b" + body + r"(?:s|es|ed|ing)?\b")


def _prefix_regex(word: str) -> re.Pattern:
    word = normalize(word)
    body = r"\s+".join(re.escape(p) for p in word.split())
    return re.compile((r"(?:^|[\s\W])" if KANNADA.search(word) else r"\b") + body)


CLAUSE = re.compile(r"[.;!?\n,]|\bbut\b|\band\b")

_COMPILED = None


def _compiled_lexicon():
    global _COMPILED
    if _COMPILED is None:
        lex = knowledge.lexicon()
        subs = [(sid, lang, term, _term_regex(term, lang))
                for sid, terms in lex["subcategories"].items()
                for lang, words in terms.items() for term in words]
        oos = [(kind, lang, term, _term_regex(term, lang))
               for kind, terms in lex["out_of_scope"].items()
               for lang, words in terms.items() for term in words]
        sev = [(lang, term, _term_regex(term, lang))
               for lang, words in lex["severity_cues"].items() for term in words]
        pairs = [(sid, [_prefix_regex(w) for w in rule["assets"]], [_prefix_regex(w) for w in rule["defects"]])
                 for sid, rule in lex.get("asset_defect_pairs", {}).items() if not sid.startswith("_")]
        _COMPILED = (subs, oos, sev, pairs)
    return _COMPILED


def _landmark_spans(text: str) -> list[tuple[int, int]]:
    return [m.span(1) for rx in (LANDMARK, LANDMARK_TL) for m in rx.finditer(text)]


def _match_terms(texts: list[str]) -> tuple[list[TermMatch], list[float], list[str], list[str]]:
    """Lexicon matches. A term found only inside a landmark phrase ("near the
    bus stop", "school hattira") describes the location, not the problem, so
    it gets a low weight instead of competing with the actual complaint."""
    subs, oos, sev, pairs = _compiled_lexicon()
    spans = [_landmark_spans(t) for t in texts]
    matches, weights, seen = [], [], set()
    for sid, lang, term, rx in subs:
        if (sid, term) in seen:
            continue
        hits = [(i, m.span()) for i, t in enumerate(texts) for m in rx.finditer(t)]
        if not hits:
            continue
        seen.add((sid, term))
        in_landmark = all(any(a <= s0 and e0 <= b + 1 for a, b in spans[i]) for i, (s0, e0) in hits)
        matches.append(TermMatch(subcategory_id=sid, term=term, language=lang))
        weights.append(0.25 if in_landmark else 1.0)
    # Asset + defect in the same clause ("the drain near my house is completely blocked")
    clauses = [c for t in texts for c in CLAUSE.split(t) if c.strip()]
    for sid, assets, defects in pairs:
        for clause in clauses:
            a = next((rx for rx in assets if rx.search(clause)), None)
            d = next((rx for rx in defects if rx.search(clause)), None)
            if a and d:
                if (sid, "pair") not in seen:
                    seen.add((sid, "pair"))
                    matches.append(TermMatch(subcategory_id=sid, term="asset+defect: " + clause.strip()[:60], language="pair"))
                    weights.append(1.0)
                break
    out_signals = sorted({f"{kind}:{term}" for kind, lang, term, rx in oos if any(rx.search(t) for t in texts)})
    severity = sorted({term for lang, term, rx in sev if any(rx.search(t) for t in texts)})
    return matches, weights, out_signals, severity


def _score(matches: list[TermMatch], match_weights: list[float]) -> list[TextCandidate]:
    weights: dict[str, float] = {}
    for m, base in zip(matches, match_weights):
        # Multi-word phrases are more specific than single words
        specificity = 1.5 if m.language == "pair" else 1.0 + 0.5 * (len(m.term.split()) - 1)
        weights[m.subcategory_id] = weights.get(m.subcategory_id, 0.0) + base * specificity
    # A specific reading overrides a generic one it implies ("sewage water is leaking"
    # is a sewage leak, not a drinking-water pipe leak)
    for strong, weak_ids in knowledge.lexicon().get("suppresses", {}).items():
        if strong in weights:
            for weak in weak_ids:
                if weak in weights:
                    weights[weak] *= 0.3
    cands = [TextCandidate(subcategory_id=s, score=round(1 - math.exp(-1.2 * w), 3)) for s, w in weights.items()]
    return sorted(cands, key=lambda c: (-c.score, c.subcategory_id))


def _locations(texts: list[str]) -> LocationMentions:
    gaz = knowledge.gazetteer()
    joined = " \n ".join(texts)
    loc = LocationMentions()
    m = PIN.search(joined)
    if m:
        loc.pin_code = m.group(1) + m.group(2)

    def has(term: str) -> bool:
        t = normalize(term)
        if KANNADA.search(t):
            return t in joined
        return re.search(r"\b" + re.escape(t) + r"\b", joined) is not None

    for c in gaz["corporation_cities"]:
        if any(has(a) for a in [c["city"], c["kn"], *c["aliases"]]):
            loc.city, loc.district = c["city"], c["district"]
            break
    localities = [l for l in gaz["bengaluru_localities"] if has(l)]
    if localities:
        loc.localities = localities
        loc.city = loc.city or "Bengaluru"
        loc.district = loc.district or "Bengaluru Urban"
    if not loc.district:
        for d in gaz["districts"]:
            if any(has(a) for a in [d["name"], d["kn"], *d["aliases"]]):
                loc.district = d["name"]
                break
    rural = gaz["rural_markers"]
    if any(has(t) for lang in rural for t in rural[lang]):
        loc.rural = True
    for road_class, markers in gaz["road_class_markers"].items():
        if any(normalize(mk) in joined for mk in markers):
            if road_class in ("national_highway", "state_highway"):
                loc.road_class = loc.road_class or road_class
            else:
                loc.asset_markers.append(road_class)
    loc.landmarks = sorted({g.strip(" .,") for rx in (LANDMARK, LANDMARK_TL) for g in rx.findall(joined)})[:5]
    return loc


def _duration(texts: list[str]) -> str | None:
    for t in texts:
        for rx in DURATION_PATTERNS:
            m = rx.search(t)
            if m:
                return m.group(1).strip()
    return None


def analyze_text(text: str, translation_en: str | None = None, translation_source: str | None = None) -> TextAnalysis:
    original = (text or "").strip()
    if not original:
        return TextAnalysis(status=StageStatus.no_input)
    norm = normalize(original)
    texts = [norm] + ([normalize(translation_en)] if translation_en else [])
    matches, match_weights, oos, severity = _match_terms(texts)
    return TextAnalysis(
        status=StageStatus.ok,
        language=detect_language(norm),
        original_text=original,
        normalized_text=norm,
        translation_en=translation_en,
        translation_source=translation_source,
        matches=matches,
        candidates=_score(matches, match_weights),
        out_of_scope_signals=oos,
        severity_cues=severity,
        duration=_duration(texts),
        location=_locations(texts),
    )


# The LLM's reading counts somewhat less than a lexicon match (it can over-read a
# vague text), but agreement between the two raises the score.
LLM_WEIGHT = 0.85
AGREEMENT_BONUS = 0.1


def merge_llm_reading(t: TextAnalysis, reading: dict | None) -> TextAnalysis:
    """Add the local LLM's zero-shot reading to the lexicon result (stage 2)."""
    if not reading or t.status != StageStatus.ok:
        return t
    t.llm_reading = reading
    cands = {c.subcategory_id: c for c in t.candidates}
    for p in reading["problems"]:
        llm_score = round(p["confidence"] * LLM_WEIGHT, 3)
        c = cands.get(p["id"])
        if c:
            c.score = round(min(1.0, max(c.score, llm_score) + AGREEMENT_BONUS), 3)
            c.sources = sorted(set(c.sources) | {"local_llm"})
        elif llm_score >= 0.15:
            cands[p["id"]] = TextCandidate(subcategory_id=p["id"], score=llm_score, sources=["local_llm"])
    t.candidates = sorted(cands.values(), key=lambda c: (-c.score, c.subcategory_id))
    # An out-of-scope reading only counts when the lexicon found no civic problem either
    if reading.get("out_of_scope") and not t.candidates:
        t.out_of_scope_signals.append(f"{reading['out_of_scope']}:llm")
    return t
