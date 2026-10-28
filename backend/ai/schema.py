"""
Structured, schema-validated output of the civic AI pipeline.

Every stage writes into one of these models; pipeline.analyze() returns an
AnalysisResult, validated by pydantic before it leaves the backend. Unknown
values are explicit nulls or 'unknown' enum members, never guesses.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class Status(str, Enum):
    ok = "ok"                                    # classified and routed; citizen still reviews the draft
    needs_citizen_input = "needs_citizen_input"  # e.g. choose primary issue, add locality
    needs_review = "needs_review"                # uncertain; officials must review
    out_of_scope = "out_of_scope"
    error = "error"


class Scope(str, Enum):
    in_scope = "in_scope"
    out_of_scope = "out_of_scope"
    uncertain = "uncertain"


class StageStatus(str, Enum):
    ok = "ok"
    no_input = "no_input"
    model_unavailable = "model_unavailable"
    invalid_input = "invalid_input"
    invalid_model_output = "invalid_model_output"
    error = "error"


class ImageQuality(str, Enum):
    good = "good"
    blurry = "blurry"
    dark = "dark"
    obstructed = "obstructed"
    unknown = "unknown"


class Language(str, Enum):
    en = "en"
    kn = "kn"
    kn_en_mixed = "kn_en_mixed"          # Kannada script mixed with English
    kn_transliterated = "kn_transliterated"  # Kannada written in Latin script
    other = "other"
    unknown = "unknown"


class Consistency(str, Enum):
    consistent = "consistent"
    partially_consistent = "partially_consistent"
    contradiction = "contradiction"                      # image shows a different civic problem
    insufficient_image_evidence = "insufficient_image_evidence"  # civic scene, reported problem not visible
    unrelated_image = "unrelated_image"                  # photo is not a civic scene at all
    image_only = "image_only"                            # no usable text
    text_only = "text_only"                              # no usable image analysis
    not_assessed = "not_assessed"


class RoutingStatus(str, Enum):
    resolved = "resolved"
    confirmation_required = "confirmation_required"
    location_required = "location_required"
    authority_not_configured = "authority_not_configured"
    not_applicable = "not_applicable"


class UrbanRural(str, Enum):
    urban = "urban"
    rural = "rural"
    unknown = "unknown"


class EvidenceSource(str, Enum):
    image_vlm = "image_vlm"
    pothole_detector = "pothole_detector"
    road_damage_detector = "road_damage_detector"
    text = "text"
    citizen_confirmed = "citizen_confirmed"


# ── Stage outputs ─────────────────────────────────────────────────────────

class VisualProblem(BaseModel):
    subcategory_id: str
    evidence: str = Field(description="What is visible in the photo that supports this problem")
    confidence: float = Field(ge=0, le=1)
    source: EvidenceSource = EvidenceSource.image_vlm


class ImageAnalysis(BaseModel):
    status: StageStatus
    images_analyzed: int = 0
    scene_description: Optional[str] = None
    is_civic_scene: Optional[bool] = None
    image_quality: ImageQuality = ImageQuality.unknown
    problems: list[VisualProblem] = []
    detector: Optional[dict[str, Any]] = None      # specialist pothole detector output, as returned
    warnings: list[str] = []
    error: Optional[str] = None


class TermMatch(BaseModel):
    subcategory_id: str
    term: str
    language: str


class TextCandidate(BaseModel):
    subcategory_id: str
    score: float = Field(ge=0, le=1)
    sources: list[str] = ["lexicon"]     # lexicon and/or local_llm


class LocationMentions(BaseModel):
    pin_code: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    localities: list[str] = []
    landmarks: list[str] = []
    rural: Optional[bool] = None
    road_class: Optional[str] = None
    asset_markers: list[str] = []


class TextAnalysis(BaseModel):
    status: StageStatus
    language: Language = Language.unknown
    original_text: str = ""
    normalized_text: str = ""
    translation_en: Optional[str] = None
    translation_source: Optional[str] = None
    matches: list[TermMatch] = []
    candidates: list[TextCandidate] = []
    out_of_scope_signals: list[str] = []
    severity_cues: list[str] = []
    duration: Optional[str] = None
    location: LocationMentions = LocationMentions()
    llm_reading: Optional[dict[str, Any]] = None   # local LLM's zero-shot reading of the text, as parsed


class Fusion(BaseModel):
    consistency: Consistency
    explanation: str
    shared_subcategories: list[str] = []
    image_only_subcategories: list[str] = []
    text_only_subcategories: list[str] = []


class Alternative(BaseModel):
    category_id: str
    subcategory_id: str
    confidence: float = Field(ge=0, le=1)
    sources: list[EvidenceSource] = []


class Classification(BaseModel):
    taxonomy_version: str
    category_id: Optional[str] = None
    subcategory_id: Optional[str] = None
    category_name_en: Optional[str] = None
    subcategory_name_en: Optional[str] = None
    subcategory_name_kn: Optional[str] = None
    confidence: float = Field(ge=0, le=1, default=0.0)
    sources: list[EvidenceSource] = []
    alternatives: list[Alternative] = []
    multiple_issues: bool = False
    primary_selection_required: bool = False
    out_of_scope: bool = False
    out_of_scope_reason: Optional[str] = None
    severity: Optional[int] = Field(default=None, ge=1, le=3)
    human_review_required: bool = False
    review_reasons: list[str] = []


class AuthorityAlternative(BaseModel):
    authority_id: str
    authority_name: str
    reason: str


class Jurisdiction(BaseModel):
    city: Optional[str] = None
    district: Optional[str] = None
    urban_rural: UrbanRural = UrbanRural.unknown
    pin_code: Optional[str] = None
    basis: list[str] = []          # how the place was determined (text, PIN, GPS approx, citizen)
    approximate: bool = True


class Routing(BaseModel):
    registry_version: str
    status: RoutingStatus
    authority_id: Optional[str] = None
    authority_name: Optional[str] = None
    authority_name_kn: Optional[str] = None
    authority_type: Optional[str] = None
    jurisdiction: Jurisdiction = Jurisdiction()
    verified_url: Optional[str] = None
    url_verification: Optional[str] = None
    authority_verification: Optional[str] = None
    routing_rationale: list[str] = []
    confidence: float = Field(ge=0, le=1, default=0.0)
    alternatives: list[AuthorityAlternative] = []
    jurisdiction_confirmation_required: bool = True
    missing_location_fields: list[str] = []
    escalation_authority_id: Optional[str] = None


class ComplaintDraft(BaseModel):
    language: str
    subject: str
    body: str
    missing_details: list[str] = []
    attachment_refs: list[str] = []
    editable: bool = True
    generated_by: str = "template"


class ModelMetadata(BaseModel):
    pipeline_version: str
    schema_version: str = SCHEMA_VERSION
    offline_mode: bool
    taxonomy_version: str
    registry_version: str
    lexicon_version: str
    vlm: dict[str, Any] = {}
    detector: Optional[str] = None
    authenticity_model: Optional[str] = None
    timings_ms: dict[str, float] = {}


class AnalysisResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    analysis_id: str
    status: Status
    scope: Scope
    image_analysis: ImageAnalysis
    text_analysis: TextAnalysis
    fusion: Fusion
    classification: Classification
    routing: Routing
    complaint_draft: Optional[ComplaintDraft] = None
    authenticity: Optional[dict[str, Any]] = None   # existing fake-image pipeline output, unmodified
    model_metadata: ModelMetadata
    errors: list[str] = []
