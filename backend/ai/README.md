# JanaSutra civic AI pipeline

> The AI identifies the civic problem; the verified registry determines which
> named Karnataka civic authority is responsible; and the citizen reviews the
> resulting complaint before submission.

Everything here runs **locally, with open-weight models, in strict offline
mode**. No complaint text or photo is sent to an external API.

## Stages

| # | Stage | Module | What it does |
|---|-------|--------|--------------|
| 1 | Image understanding | `vision.py` | Qwen3-VL-2B-Instruct describes the photo and lists visible problems as taxonomy IDs (strict JSON, validated, one retry). The existing YOLO11n pothole detector adds its own evidence, tagged `pothole_detector`. |
| 2 | Text understanding | `text_analysis.py` | Normalisation (NFC, ZWJ/ZWNJ), language detection (`en`, `kn`, `kn_en_mixed`, `kn_transliterated`), a multilingual lexicon (English, Kannada script, transliterated Kannada), location, duration and severity cues. When the VLM is installed, it also translates Kannada text. A zero-shot reading of the text (`read_text`) exists but is **off by default**; see Evaluation. |
| 3 | Consistency | `fusion.assess_consistency` | Compares what the photo shows with what the citizen says: `consistent`, `partially_consistent`, `contradiction`, `insufficient_image_evidence`, `unrelated_image`, `image_only`, `text_only`. |
| 4 | Classification | `fusion.classify` | Combines the evidence against the versioned taxonomy and returns the primary issue, alternatives, multiple-issue / primary-selection flags, out-of-scope status and review reasons. |
| 5 | Routing | `routing_engine.py` | Deterministic rules over the authority registry and gazetteer. The model never names an authority. |
| 6 | Drafting | `drafting.py` | Template draft in English or Kannada. Automated visual findings are kept separate from the citizen's quoted words, and missing details are listed. |
| 7 | Confidence and review | `pipeline.py` | Overall `status`: `ok`, `needs_citizen_input`, `needs_review`, `out_of_scope` or `error`. Also `scope`, per-stage timings and the full metadata. |

**Photo authenticity is not decided here.** The existing fake-image detector
(`services/image_authenticity.py`) runs unchanged and its output is attached
as `authenticity`. A flagged photo adds a review reason. It never changes what
the photo is judged to show, and a genuine photo never proves the text true.

### Endpoints

- `POST /api/ai/analyze` (multipart): `description`, `images[]`, location fields, `confirmed_subcategory_id`, `city_corporation_id`, `draft_language`. Returns the schema in `schema.py`. Pass `analysis_id` to re-run stages 3–7 after a citizen choice without running the models again.
- `POST /api/complaints` accepts the same `analysis_id` plus `confirmed_subcategory_id`, `confirmed_authority_id`, `city_corporation_id` and `complaint_text` (the edited draft). Out-of-scope requests get HTTP 422. An authority that does not handle the problem gets HTTP 400. The full result is stored in `ai_analysis.civic_ai`.
- `GET /api/ai/taxonomy`, `GET /api/ai/authorities`, `GET /api/ai/status` (installed and loaded models; no inference).

### Status rules (deliberate)

- **Out of scope only when nothing civic is supported.** The text names no civic problem, no photo evidence shows one, and either the text matches a non-civic signal (pension, police, landlord…) or the photo is not a public scene.
- **Mismatch is never treated as out of scope.** A contradiction, unrelated photo or insufficient evidence becomes a review reason, and the text-only claim gets a lower score.
- **Several distinct problems with close scores** set `primary_selection_required`, and the citizen chooses.
- **Bengaluru** is split across five GBA city corporations, and no verified ward boundaries exist offline, so the citizen picks the corporation. The legacy name BBMP is an alias only, never a destination.

## Models

| Model | Licence | Use | Where |
|-------|---------|-----|-------|
| Qwen/Qwen3-VL-2B-Instruct @ `89644892e4d85e24eaac8bacfd4f463576704203` | Apache-2.0 | photo understanding, Kannada→English translation, zero-shot text reading | `models/qwen3-vl-2b-instruct` (4.26 GB bf16) |
| YOLO11n pothole detector (trained in this project, Ultralytics) | AGPL-3.0 (Ultralytics) | pothole evidence | `backend/ml_models/pothole.onnx` |
| EfficientNet-B0 authenticity classifier (trained in this project from torchvision ImageNet weights) | project-trained; backbone weights under torchvision's terms | fake / screen photo flag | `backend/ml_models/authenticity.onnx` |

**Why Qwen3-VL-2B.** It is the largest open-weight VLM with a permissive
licence that fits an 8 GB machine. The alternatives were rejected:

- Qwen2.5-VL-3B: its Qwen Research licence is non-commercial.
- 7B and 8B VLMs: too large for 8 GB RAM, and impossible on the 512 MB API host.
- Text-only models: they cannot check the photo.

Its Kannada ability is limited at this size. That is why the deterministic
lexicon always runs and the model's reading is merged as extra evidence
rather than trusted alone.

### Install / update (the only networked step)

```bash
python scripts/install_models.py            # downloads the pinned revision, verifies SHA-256, writes models/MODEL_MANIFEST.json
python scripts/install_models.py --verify   # offline re-check of every file against the pinned hashes
```

To update, change `revision` and the `model.safetensors` hash in
`scripts/install_models.py`, re-run it, then re-run the evaluation below
before deploying.

### Configuration (`backend/config.py`, env vars)

| Setting | Default | Notes |
|---------|---------|-------|
| `AI_STRICT_OFFLINE` | `true` | sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`; models load with `local_files_only=True` |
| `AI_VLM_ENABLED` | `true` | set `false` on small hosts; the pipeline then runs lexicon + detectors only and reports `model_unavailable` |
| `AI_VLM_PATH` | `models/qwen3-vl-2b-instruct` | |
| `AI_VLM_DEVICE` | `auto` | `cuda` → `mps` → `cpu` |
| `AI_VLM_DTYPE` | `auto` | bf16 on MPS and CUDA (fp16 if no bf16), fp32 on CPU |
| `AI_VLM_QUANTIZATION` | `none` | `bnb-8bit` / `bnb-4bit` (CUDA + bitsandbytes), `dynamic-int8` (CPU only) |
| `AI_VLM_MAX_IMAGE_SIDE` | 640 | larger photos are downscaled before the VLM (the evaluated setting; 896 was ~40% slower on a 6-image check) |
| `AI_VLM_TRANSLATE_KANNADA` | `true` | Kannada → English translation by the VLM, fed to the lexicon |
| `AI_LLM_TEXT_CLASSIFY` | `false` | zero-shot text reading by the VLM; off because it labels non-civic requests as potholes (measured below) |
| `AI_REVIEW_CONFIDENCE` | 0.6 | below this, classification needs review |

**Deployment note.** The free Render instance (512 MB) cannot hold the VLM,
so production runs with `AI_VLM_ENABLED=false`: lexicon, pothole detector and
authenticity model only. Running the full pipeline needs about 6 GB of free
RAM, or a GPU with about 5 GB (about 2.5 GB with `bnb-4bit`).

### Verifying offline operation

1. `python -m pytest backend/tests/test_offline.py`. This blocks every socket call and runs the full pipeline: stubbed VLM, real ONNX detectors, translation, drafting.
2. Real model, no network: switch Wi-Fi off and run `AI_STRICT_OFFLINE=true python -c "from backend.ai import vision; vision.load(); print(vision.metadata())"`.
3. `GET /api/ai/status` reports `offline_mode`, the model path, and whether the model is installed and loaded.

## Knowledge files (`data/`, versioned)

| File | Version | Content |
|------|---------|---------|
| `taxonomy.json` | 1.1.0 | 13 categories, 44 subcategories (incl. 3 out-of-scope), stable IDs (`ROAD.POTHOLE`…), English and Kannada names, severity, legacy code |
| `authority_registry.json` | 1.0.0 | 39 authority entries (incl. 10 city corporations and 2 placeholders) |
| `lexicon.json` | 1.2.0 | per-subcategory terms (en / kn / transliterated), asset+defect pairs, out-of-scope signals, suppression rules |
| `gazetteer.json` | 1.0.0 | 31 districts, corporation cities, Bengaluru localities, road-class markers |

Rules: IDs are never renamed or reused; adding IDs bumps the minor version.

**Registry provenance (compiled 2026-10-28).** An `official_source_url` is
present only when the URL returned HTTP 200 and its page title named the
authority (18 entries, `url_verified`). 4 candidate URLs did not respond and
are withheld (`unreachable`). 17 entries have no URL (`unverified`). Only one
service-area fact was read from an official site: CESC's five districts. All
other jurisdictions are marked `unverified`, and the routing engine asks for
confirmation whenever it relies on one. No complaint-submission URLs or phone
numbers are included unless they were checked (BWSSB helpline 1916).
`python scripts/verify_registry.py` re-checks every URL and writes a report.
A person then updates entries and bumps the version.

## Evaluation (measured, not estimated)

`python training/civic_ai/evaluate.py <set.jsonl> [--llm]` reports macro-F1,
per-class precision/recall, out-of-scope precision/recall, routing accuracy,
language accuracy and latency.

**Both evaluation sets are developer-authored**, not real citizen complaints.
They show how the system behaves on varied phrasing, not real-world accuracy.

| Set | Mode | n | Subcat. acc. | Macro-F1 | Category acc. | OOS P / R | Routing acc. | Language acc. |
|-----|------|---|--------------|----------|---------------|-----------|--------------|---------------|
| v1 (first run, before any fixes) | lexicon only | 48 | 0.875 | 0.763 | 0.875 | 1.00 / 1.00 | 0.867 (30) | 0.976 |
| v1 (after fixing the errors it exposed; **no longer held out**) | lexicon only | 48 | 1.000 | 1.000 | 1.000 | 1.00 / 1.00 | 1.000 | 0.976 |
| **v2 held-out** (written after the fixes, never tuned on) | lexicon only | 42 | **0.524** | **0.475** | 0.571 | 1.00 / 0.50 | 0.895 (19) | 1.000 |
| v2 (used for the on/off decision below) | lexicon + Qwen3-VL-2B text reading | 42 | 0.595 | 0.511 | 0.643 | 1.00 / **0.17** | 0.842 (19) | 1.000 |

Reading the held-out result:

- The lexicon is precise but brittle on paraphrases ("tar has peeled off", "naked electric cable").
- 13 of its 20 errors fall back to `OTHER.CIVIC_INFRA` with `needs_review`, which is a safe failure. 7 are a wrong category.
- Out-of-scope requests it misses also go to review; none are wrongly filed as civic problems with high confidence.
- Text-stage latency is about 2 ms per complaint (lexicon); 6.2 s median with the LLM reading (Apple M2, MPS).
- **The LLM text reading was switched off by default because of this result.** It fixed some paraphrases, but it labelled four of six non-civic requests (caste certificate, rude doctor, property dispute, pension) as `ROAD.POTHOLE`. Filing non-civic requests as road complaints is worse than sending vague civic ones to review. v2 was used for this one on/off decision, so it is no longer fully held out for the LLM configuration.

### Image understanding (Qwen3-VL-2B, stage 1)

`python training/civic_ai/evaluate_images.py --n 30` was run with `AI_VLM_MAX_IMAGE_SIDE=640` on an Apple M2 (8 GB, MPS, bf16). The VLM ran alone, with no pothole detector. Report: `training/civic_ai/eval/vlm_images_report.json`.

| Metric | Value |
|--------|-------|
| Valid JSON (after one retry) | 0.933 (56 / 60) |
| Pothole listed on 30 held-out pothole photos | 0.367 (any `ROAD.*`: 0.40) |
| Pothole photos judged a civic scene | 0.90 |
| Non-civic photos (11 ImageNet photos) with any civic problem listed | **0.00** |
| Non-civic photos judged not a civic scene | 0.545 |
| Latency per image, median / p95 | 23.5 s / 113 s (long outputs and retries) |
| Accelerator memory | 6.4 GB (MPS); model load 16 s |

Correction recorded in the report: the first run sampled 30 "negatives", but 19 of them were road photos mixed into the authenticity "real" folder. They are excluded, and the sampler now uses only the ImageNet photos.

What this means:
- The 2B VLM is reliable at saying what a photo is **not** (no false civic problems), and it describes scenes usefully for the reviewer.
- It is weak at spotting potholes. The dedicated detector does much better on the same test split (image recall 0.77 at threshold 0.2), so the pipeline keeps both and labels each piece of evidence with its source.
- No local labelled photos exist for other categories, so garbage, drains and the rest are unmeasured.

Reports are stored next to the sets in `training/civic_ai/eval/`.

## Training pipeline (`training/civic_ai/`)

- `dataset.py`: record schema (`--schema`) and validation. It excludes unreviewed and developer-authored records. De-duplication removes exact duplicates, clusters near-duplicates at char-3gram cosine ≥ 0.9, and drops conflicting labels. The split is deterministic and group-aware (70/15/15): near-duplicates and records sharing a `group`, such as a ward or master ticket, never cross splits. It writes a manifest with SHA-256 hashes.
- `train_text_classifier.py`: TF-IDF char n-grams + logistic regression, with C chosen on validation and metrics on test. **It refuses to train** with fewer than 30 reviewed examples in at least 5 subcategories. Today there are none: the platform has no reviewed labelled complaints yet, so no classifier is trained or shipped.
- **LoRA fine-tuning is not justified yet.** It would need thousands of reviewed image+text complaints and evidence that the zero-shot VLM and the lightweight classifier fall short on them.

## Tests

`python -m pytest backend/tests` covers:

- Category coverage and languages: English, Kannada, transliterated and code-mixed text.
- Landmark handling, multiple issues and primary selection.
- Out of scope versus mismatch.
- Contradiction, unrelated and insufficient-evidence photos.
- Malformed and unknown VLM output, missing model, corrupt image, flagged authenticity, and detector evidence.
- Routing: Bengaluru corporations, BWSSB, CESC, NHAI, BMRCL, other corporation cities, rural panchayat, missing location, and URLs only from the registry.
- Drafting in English and Kannada, schema round-trip, determinism across hash seeds.
- The HTTP flow: analyse, re-analyse, submit, and rejection paths.
- Knowledge-file integrity, dataset split leakage, and offline execution.

## Limitations

- **No ward or panchayat boundaries.** Below city level, routing needs the citizen's choice, and a GPS-only location is approximate (corporation-city radius).
- **Registry coverage:** most jurisdictions and asset responsibilities are unverified. Individual gram panchayats and town municipal councils are not listed (`authority_not_configured`). Asset ownership (PWD vs corporation road, BWSSB vs corporation manhole, BESCOM vs corporation streetlight) often needs confirmation.
- **Lexicon recall on paraphrases is low** (v2: 0.52). The 2B LLM's text reading raised it to 0.60 but caused out-of-scope errors, so it is off. A larger local model, or a classifier trained on reviewed complaints, is the path to better recall.
- **The VLM is slow on CPU/MPS** (median 23 s per photo on an M2). The free production host cannot run it at all.
- **The 2B VLM's Kannada is weak.** Translation errors can mislead its reading, so the lexicon result is always kept alongside.
- **Drafts are templates.** They do not invent facts, but they also don't summarise long descriptions.
- **The analysis cache is in-process** (256 entries). After a restart, a submission re-runs the text stages, and the background tasks re-run the photo checks.
- **Evaluation sets are small and developer-authored.** Real accuracy needs reviewed citizen complaints (see the training pipeline).
