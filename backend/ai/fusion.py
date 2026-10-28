"""
Stages 3 and 4: image-text consistency, then civic complaint classification.

Visual evidence (VLM, pothole detector) and citizen-reported claims (text)
stay separate: the fusion step only compares them, and a genuine photo never
turns a text claim into a visual finding.
"""

from backend.ai import knowledge
from backend.ai.schema import (Alternative, Classification, Consistency, EvidenceSource, Fusion, ImageAnalysis,
                               StageStatus, TextAnalysis)
from backend.config import settings

# Generic subcategories lose to a more specific sibling found in the same input
GENERIC = {"ROAD.SURFACE_DAMAGE": "ROAD", "WASTE.UNCOLLECTED": "WASTE", "OTHER.CIVIC_INFRA": "OTHER"}
# Minimum score for a candidate to be listed / to count as a separate issue
MIN_CANDIDATE = 0.15
MULTI_ISSUE = 0.45
# Top two candidates of different categories closer than this need the citizen to choose
PRIMARY_MARGIN = 0.15


def _image_scores(image: ImageAnalysis) -> dict[str, float]:
    scores = {}
    for p in image.problems:
        scores[p.subcategory_id] = max(scores.get(p.subcategory_id, 0.0), p.confidence)
    return scores


def assess_consistency(image: ImageAnalysis, text: TextAnalysis) -> Fusion:
    img = _image_scores(image)
    txt = {c.subcategory_id: c.score for c in text.candidates}
    image_ok = image.status == StageStatus.ok
    text_ok = text.status == StageStatus.ok and bool(text.normalized_text)

    if not image_ok and not text_ok:
        return Fusion(consistency=Consistency.not_assessed, explanation="Neither a usable image analysis nor text.")
    if not image_ok:
        note = {StageStatus.no_input: "No photo was provided.",
                StageStatus.model_unavailable: "The image-understanding model is not available; only the citizen's "
                                               "text (and the pothole detector, if it ran) were used.",
                StageStatus.invalid_input: "The photo could not be read.",
                StageStatus.invalid_model_output: "The image model returned an invalid answer."}
        return Fusion(consistency=Consistency.text_only, explanation=note.get(image.status, "Image not analysed."),
                      image_only_subcategories=sorted(img), text_only_subcategories=sorted(txt))
    if not text_ok or not txt:
        return Fusion(consistency=Consistency.image_only,
                      explanation="The text names no recognisable civic problem; classification relies on the photo.",
                      image_only_subcategories=sorted(img))

    shared = sorted(set(img) & set(txt))
    img_cats = {knowledge.category_of(s) for s in img}
    txt_cats = {knowledge.category_of(s) for s in txt}
    if image.is_civic_scene is False and not img:
        return Fusion(consistency=Consistency.unrelated_image,
                      explanation="The photo does not show a street, public place or public infrastructure, "
                                  "so it cannot support the reported problem.",
                      text_only_subcategories=sorted(txt))
    if not img:
        return Fusion(consistency=Consistency.insufficient_image_evidence,
                      explanation="The photo shows a public scene but the reported problem is not clearly visible.",
                      text_only_subcategories=sorted(txt))
    if shared:
        full = set(txt) <= set(img)
        return Fusion(consistency=Consistency.consistent if full else Consistency.partially_consistent,
                      explanation="The photo shows the problem the citizen describes." if full else
                      "The photo supports part of the description; some reported details are not visible.",
                      shared_subcategories=shared, image_only_subcategories=sorted(set(img) - set(txt)),
                      text_only_subcategories=sorted(set(txt) - set(img)))
    if img_cats & txt_cats:
        return Fusion(consistency=Consistency.partially_consistent,
                      explanation="The photo and text describe related problems of the same kind.",
                      image_only_subcategories=sorted(img), text_only_subcategories=sorted(txt))
    return Fusion(consistency=Consistency.contradiction,
                  explanation="The photo shows a different civic problem from the one described.",
                  image_only_subcategories=sorted(img), text_only_subcategories=sorted(txt))


def classify(image: ImageAnalysis, text: TextAnalysis, fusion: Fusion,
             confirmed_subcategory: str | None = None) -> Classification:
    tax_version = knowledge.taxonomy()["version"]
    img = _image_scores(image)
    txt = {c.subcategory_id: c.score for c in text.candidates}
    found_by = {}
    for p in image.problems:
        found_by.setdefault(p.subcategory_id, set()).add(p.source)

    # Combined score: average of the evidence that exists; agreement between
    # image and text raises the score, a one-sided claim is capped lower.
    combined, sources = {}, {}
    for sid in sorted(set(img) | set(txt)):
        i, t = img.get(sid), txt.get(sid)
        if i is not None and t is not None:
            score = min(1.0, 0.5 * (i + t) + 0.2)
        elif i is not None:
            score = i * (0.85 if text.status == StageStatus.ok and txt else 0.95)
        else:
            # A text claim the photo does not support counts much less than an unchecked one
            score = t * (0.6 if fusion.consistency in (Consistency.contradiction, Consistency.unrelated_image,
                                                       Consistency.insufficient_image_evidence) else 0.9)
        combined[sid] = round(score, 3)
        src = [s for s in (EvidenceSource.image_vlm, EvidenceSource.pothole_detector,
                           EvidenceSource.road_damage_detector) if s in found_by.get(sid, ())]
        if t is not None:
            src.append(EvidenceSource.text)
        sources[sid] = src
    # Prefer a specific subcategory over its generic sibling; the specific one
    # inherits the generic one's evidence (photo shows "garbage", text says "dumped")
    for generic, cat in GENERIC.items():
        siblings = [s for s in combined if s != generic and knowledge.category_of(s) == cat]
        if generic in combined and siblings:
            for s in siblings:
                if combined[generic] > combined[s]:
                    combined[s] = combined[generic]
                    sources[s] = sorted(set(sources[s]) | set(sources[generic]), key=lambda e: e.value)
            combined[generic] = round(combined[generic] * 0.6, 3)

    review = []
    oos_signals = text.out_of_scope_signals
    civic_text = bool(txt)
    image_civic = image.status == StageStatus.ok and image.is_civic_scene is not False
    image_has_problem = bool(img)

    # Out of scope only when nothing civic is supported by text or image
    if not civic_text and not image_has_problem and confirmed_subcategory is None:
        reason, sub = None, None
        if oos_signals:
            kind = oos_signals[0].split(":", 1)[0]
            sub = "OUT_OF_SCOPE." + kind
            terms = [s.split(":", 1)[1] for s in oos_signals if not s.endswith(":llm")]
            reason = "The request concerns " + ("a personal or private dispute" if kind == "PERSONAL_DISPUTE"
                                                else "a non-civic government or private service matter") + \
                     (f" ({', '.join(terms)})" if terms else " (as read by the local language model)") + \
                     ", not public infrastructure."
        elif image.status == StageStatus.ok and image.is_civic_scene is False:
            sub = "OUT_OF_SCOPE.UNRELATED_IMAGE"
            reason = "The photo is not a civic scene and the text names no civic infrastructure problem."
        if sub:
            s = knowledge.subcategory(sub)
            return Classification(taxonomy_version=tax_version, category_id="OUT_OF_SCOPE", subcategory_id=sub,
                                  category_name_en=s["category_name_en"], subcategory_name_en=s["name_en"],
                                  subcategory_name_kn=s["name_kn"], confidence=0.8 if oos_signals else 0.7,
                                  out_of_scope=True, out_of_scope_reason=reason,
                                  human_review_required=not oos_signals,
                                  review_reasons=[] if oos_signals else ["Unrelated photo with no civic description"])

    if confirmed_subcategory:
        if not knowledge.subcategory(confirmed_subcategory) or confirmed_subcategory.startswith("OUT_OF_SCOPE."):
            raise ValueError(f"Unknown or out-of-scope subcategory {confirmed_subcategory!r}")
        if confirmed_subcategory not in combined:
            review.append("Citizen selected a category not detected in the photo or text")
        combined[confirmed_subcategory] = max(combined.get(confirmed_subcategory, 0.0), 0.9)
        sources.setdefault(confirmed_subcategory, []).append(EvidenceSource.citizen_confirmed)

    # Ties: the more severe problem first, then the ID, so results never depend on set order
    ranked = sorted(combined.items(), key=lambda kv: (-kv[1], -knowledge.subcategory(kv[0])["severity"], kv[0]))
    if confirmed_subcategory:
        ranked = [(confirmed_subcategory, combined[confirmed_subcategory])] + \
                 [kv for kv in ranked if kv[0] != confirmed_subcategory]
    if not ranked:
        # Civic-sounding but unclassifiable (e.g. vague text, no image model)
        s = knowledge.subcategory("OTHER.CIVIC_INFRA")
        return Classification(taxonomy_version=tax_version, category_id="OTHER", subcategory_id="OTHER.CIVIC_INFRA",
                              category_name_en=s["category_name_en"], subcategory_name_en=s["name_en"],
                              subcategory_name_kn=s["name_kn"], confidence=0.2, severity=s["severity"],
                              human_review_required=True,
                              review_reasons=["No recognisable civic problem in the text or photo"])

    top_id, top_score = ranked[0]
    alternatives = [Alternative(category_id=knowledge.category_of(s), subcategory_id=s, confidence=sc,
                                sources=sources.get(s, [])) for s, sc in ranked[1:] if sc >= MIN_CANDIDATE][:5]
    distinct_cats = {knowledge.category_of(s) for s, sc in ranked if sc >= MULTI_ISSUE}
    multiple = len(distinct_cats) > 1
    primary_needed = (not confirmed_subcategory and len(ranked) > 1
                      and knowledge.category_of(ranked[1][0]) != knowledge.category_of(top_id)
                      and ranked[1][1] >= MULTI_ISSUE and top_score - ranked[1][1] < PRIMARY_MARGIN)

    if top_score < settings.AI_REVIEW_CONFIDENCE:
        review.append(f"Low classification confidence ({top_score:.2f})")
    if primary_needed:
        review.append("Several different problems detected; the citizen should choose the primary issue")
    if fusion.consistency in (Consistency.contradiction, Consistency.unrelated_image,
                              Consistency.insufficient_image_evidence):
        review.append(f"Image-text check: {fusion.consistency.value.replace('_', ' ')}")
    if oos_signals and not civic_text:
        review.append("The description concerns a non-civic matter (" +
                      ", ".join(x.split(":", 1)[1] for x in oos_signals) +
                      "); only the photo shows a civic problem")
    if image.status == StageStatus.ok and image.image_quality in ("blurry", "dark", "obstructed"):
        review.append(f"Photo quality: {image.image_quality.value}")

    sub = knowledge.subcategory(top_id)
    severity = sub["severity"]
    if text.severity_cues and severity < 3:
        severity += 1
    return Classification(
        taxonomy_version=tax_version,
        category_id=sub["category_id"], subcategory_id=top_id,
        category_name_en=sub["category_name_en"], subcategory_name_en=sub["name_en"],
        subcategory_name_kn=sub["name_kn"],
        confidence=round(top_score, 3), sources=sources.get(top_id, []),
        alternatives=alternatives, multiple_issues=multiple, primary_selection_required=primary_needed,
        severity=severity, human_review_required=bool(review), review_reasons=review,
    )
