"""
AI Service: connects complaint submission to the civic AI pipeline
(backend/ai). The pipeline classifies against the versioned taxonomy and
routes through the verified authority registry; this module maps that result
onto the legacy complaint columns (category / severity / department) and keeps
the full structured result in ai_analysis["civic_ai"].
"""

import re
import zlib

from backend.ai import knowledge, pipeline
from backend.ai.routing_engine import CitizenLocation
from backend.ai.schema import AnalysisResult, RoutingStatus

PENDING_AUTHORITY = "Pending authority confirmation"


class OutOfScopeComplaint(ValueError):
    """The request is not a civic infrastructure complaint (e.g. a personal dispute)."""


def _department(result: AnalysisResult, confirmed_authority_id: str | None,
                citizen: CitizenLocation) -> tuple[str, str | None, str]:
    """(department label, authority_id, basis). The registry, not the model, names the authority."""
    routing = result.routing
    if confirmed_authority_id:
        suggested = {routing.authority_id} | {a.authority_id for a in routing.alternatives}
        sub = result.classification.subcategory_id
        if confirmed_authority_id not in suggested and not knowledge.handles(confirmed_authority_id, sub):
            raise ValueError(f"Authority {confirmed_authority_id} does not handle {sub}")
        return knowledge.authority(confirmed_authority_id)["official_name"], confirmed_authority_id, "citizen_confirmed"
    if routing.authority_id and routing.status in (RoutingStatus.resolved, RoutingStatus.confirmation_required,
                                                   RoutingStatus.authority_not_configured):
        if routing.status == RoutingStatus.resolved:
            basis = "registry"
        elif routing.authority_id in (citizen.city_corporation_id, citizen.asset_owner_id):
            basis = "citizen_confirmed"
        else:
            basis = "registry_unconfirmed"
        return routing.authority_name, routing.authority_id, basis
    return PENDING_AUTHORITY, None, "unresolved"


def analyze_complaint(description: str, images: list[bytes] | None = None, citizen: CitizenLocation | None = None,
                      analysis_id: str | None = None, confirmed_subcategory: str | None = None,
                      confirmed_authority_id: str | None = None, submitted_draft: str | None = None) -> dict:
    """Classify and route a complaint being submitted.

    Reuses the citizen's reviewed analysis (analysis_id from POST /api/ai/analyze)
    when it is still cached, re-running only classification and routing with the
    citizen's confirmations. Otherwise runs the text stages now; the photo
    detectors run afterwards as background tasks.

    Raises OutOfScopeComplaint for requests the platform should not file, and
    ValueError for an authority that does not handle the confirmed problem."""
    images = images or []
    citizen = citizen or CitizenLocation()
    prior = pipeline.cached(analysis_id)
    result = pipeline.analyze(images, description, citizen, confirmed_subcategory=confirmed_subcategory,
                              run_vlm=False, run_detectors=False, prior=prior)
    if result.classification.out_of_scope:
        raise OutOfScopeComplaint(result.classification.out_of_scope_reason or "Not a civic infrastructure problem")

    department, authority_id, basis = _department(result, confirmed_authority_id, citizen)
    sub = knowledge.subcategory(result.classification.subcategory_id)
    civic = result.model_dump(mode="json")
    civic["submission"] = {
        "reused_analysis_id": prior.analysis_id if prior else None,
        "confirmed_subcategory_id": confirmed_subcategory,
        "confirmed_authority_id": confirmed_authority_id,
        "department_basis": basis,
        "authority_id": authority_id,
        "citizen_location": {k: v for k, v in vars(citizen).items() if v is not None},
        "submitted_draft": submitted_draft,
    }
    return {
        "category": sub["legacy_code"],
        "subcategory_id": result.classification.subcategory_id,
        "severity": result.classification.severity or sub["severity"],
        "department": department,
        "description": f"Detected: {sub['name_en']} ({result.classification.confidence:.0%} confidence).",
        # Decided by the fake-image detector: from the reviewed analysis when it
        # ran, otherwise by the verify_authenticity background task.
        "is_authentic": not (result.authenticity or {}).get("flagged", False),
        "civic_ai": civic,
    }


# ── Text embedding stubs (for deduplication) ─────────────────────────────

EMBEDDING_DIM = 384


def get_text_embedding(text: str) -> list[float]:
    """
    384-dim hashed character 3-gram vector (L2-normalised) for deduplication.
    Character n-grams tolerate typos, transliteration and mixed-language text
    ("pothole"/"pot hole"/"potholes") without loading an ML model; IndicSBERT
    can replace this later behind the same signature.
    """
    words = re.sub(r"[^\w\s]", " ", (text or "").lower()).split()
    vec = [0.0] * EMBEDDING_DIM
    for word in words:
        padded = f" {word} "
        for i in range(len(padded) - 2):
            # crc32 is stable across processes (built-in hash() is randomised per run)
            vec[zlib.crc32(padded[i:i + 3].encode()) % EMBEDDING_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


def compute_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a * a for a in vec1) ** 0.5
    norm_b = sum(b * b for b in vec2) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)
