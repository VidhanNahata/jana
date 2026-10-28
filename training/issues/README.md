# All-category issue detection

Extends JanaSutra beyond potholes. Two optional models feed the existing pipeline as extra
visual evidence (`issue_classifier` / `issue_detector` sources in `backend/ai/schema.py`):

| Model | Answers | Output | Categories |
|-------|---------|--------|------------|
| **Photo classifier** (CLIP + small head, or EfficientNet-B0) | "What problem does this photo show?" | one label per photo | pothole, garbage, drain/sewage, waterlogging, streetlight, fallen tree, footpath damage, damaged sign, or `background` |
| **Box detector** (YOLO11) | "Where is it?" | boxes | garbage, open manhole, pothole |

**Recommended route for a small dataset (~150 photos per category): the CLIP option** (section 2).
It reuses a large pretrained image model and trains only a tiny head, so it needs far fewer photos
and trains in seconds. The EfficientNet fine-tune (section 3) needs several hundred per class.

Potholes are still decided by the dedicated `pothole.onnx` detector; the new models never
override it. If neither new model is installed the app behaves exactly as before.

> **Relation to `backend/services/civic_scene.py`.** The repo already ships a *zero-shot* CLIP scene
> classifier (fallen tree, garbage, flooding, debris, open manhole; prompts in `training/clip_scene_prompts.py`,
> measured by `training/eval_civic_detector.py`). It needs no photos. The models in this folder are
> *trained on your photos* and add the categories it lacks (drains/sewage, streetlight poles, footpath damage,
> damaged signs) and a "normal street" class. Both feed the same pipeline as separate evidence sources
> (`scene_classifier` vs `issue_classifier`); fusion keeps the strongest score per subcategory, so running both
> does not double-count. Garbage maps to `WASTE.DUMPING` in the scene classifier and `WASTE.UNCOLLECTED` here:
> edit `classes.json` if you want them aligned. Train your head, compare its test numbers with
> `eval_civic_detector.py` on the same photos, and keep whichever is better per category.

Class names and their taxonomy mapping live in one file: `classes.json`.

## 1. Get data (the part that decides quality)

Nothing here can be trained without photos. Starting points (**check each licence and
class labels yourself; I could not verify them from the build environment**):

| Category | Candidates found by search |
|----------|----------------------------|
| garbage | [TACO](https://universe.roboflow.com/yangguangs-workspace/taco-trash-annotations-in-context-u6giq) (Roboflow copy of the TACO litter dataset), [WOLWES](https://universe.roboflow.com/pler/wolwes), [Hugging Face trash_detection](https://huggingface.co/datasets/Akhila-9849/trash_detection) |
| drain / sewage / manholes | [S-BIRD sewer imagery](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10056167/) (inside sewers, not street scenes: only useful for `drain_sewage` if you accept that) |
| waterlogging | [Urban waterlogging benchmark](https://arxiv.org/pdf/2407.08109), [Flood Master DB](https://aiia.csd.auth.gr/?p=7359) |
| footpath / road assets / signs | [Damaged Road Assets](https://universe.roboflow.com/dennas-workspace/damaged-road-assets), [UrbanProblems](https://universe.roboflow.com/sih25-8wj8v/urbanproblems-cfwk2), [road-issues-detection (HF)](https://huggingface.co/datasets/Programmer-RD-AI/road-issues-detection-dataset), [RSD-BD](https://data.mendeley.com/datasets/745n5wnt7v) |
| pothole | already in `training/data/potholes` |

Expect gaps. Public data for **streetlights, fallen trees and open manholes in Indian
streets is thin**: plan to add your own photos (the app's own reports, once reviewed, are the
best source). Train only the classes you have at least a few hundred varied photos for, and
look at the per-class numbers before trusting any of them.

`background` matters as much as any issue: put ordinary streets, shops, people, vehicles and
non-civic photos there, or the classifier will see a problem in everything.

**How many photos?** About 150 per category (and 150 `background`) is a good target; 100 is the
minimum that gives usable results. `prepare_classifier.py` holds out at least 30 test photos per class,
so with 100 it can only give ~27 (it warns you), and with 150 you get 30 test / 15 val / ~105 train.
Test scores from fewer than 30 photos per class swing wildly: don't trust them.

## 2. CLIP option (recommended)

```bash
pip install -r requirements_ai.txt huggingface_hub
python scripts/install_clip.py                          # one-off download (~600 MB), the only networked step
python training/issues/prepare_classifier.py            # photos in training/data/issues_raw/<class>/
python training/issues/train_clip_head.py               # embeds photos, fits the head (minutes), prints a report
python training/issues/train_clip_head.py --install     # same, and installs backend/ml_models/issue_clip_head.*
```

The report compares your fitted head with a zero-shot baseline (CLIP matching text prompts, no training),
so you can see what your photos add. Fix weak classes by adding 30-50 photos each, re-running `prepare_classifier.py`, and re-running
the head training. Embeddings are cached per split and recomputed automatically when the photo list
changes (so a re-run after adding photos re-embeds that split).

The backend loads CLIP strictly offline from `models/clip-vit-base-patch32` (`ISSUE_CLIP_MODEL_PATH`) and
needs `torch` + `transformers` (already in `requirements_ai.txt`). It runs in-process on the first photo;
a missing or broken encoder is logged and skipped, never fails a complaint.

## 3. EfficientNet classifier (needs more photos)

```bash
# photos go in training/data/issues_raw/<class>/  (folder names = keys in classes.json + background)
python training/issues/prepare_classifier.py     # dedupes near-copies, splits 80/10/10 by group
python training/issues/train_classifier.py --epochs 15
python training/issues/evaluate_classifier.py    # per-class thresholds from VAL, metrics on TEST
python training/issues/export_classifier.py      # -> backend/ml_models/issue_classifier.{onnx,json}
```

`evaluate_classifier.py` flags any class with fewer than 30 test photos as unreliable and
reports the `background_false_issue_rate` (how often a normal photo would wrongly raise an
issue). Do not ship a class whose test precision you wouldn't be happy to explain.

## 4. Detector (optional, for boxes)

```bash
cp training/issues/detector_sources.example.json training/issues/detector_sources.json   # edit
# unpack YOLO-format exports under training/data/raw_det/<name>/
python training/issues/prepare_detector.py
python training/train.py --model yolo11n.pt --data training/data/issues_det/data.yaml --name issues_yolo11n --epochs 80
python training/issues/export_detector.py training/runs/issues_yolo11n/weights/best.pt
```

## 5. Check it is live

`GET /api/ai/status` now has an `issue_models` block (`installed: true/false` for `clip`, `classifier`, `detector`).
`POST /api/ai/analyze` returns each finding in `image_analysis.problems` with its `source`.

Settings (env vars): `ISSUE_MODELS_ENABLED`, `ISSUE_CLIP_MODEL_PATH`, `ISSUE_CLIP_HEAD_PATH`, `ISSUE_CLASSIFIER_PATH`, `ISSUE_DETECTOR_PATH`,
`ISSUE_MODEL_MAX_CONFIDENCE` (default 0.85: one model alone never claims more).

## Not done yet

- **Browser (live camera) detection** only covers potholes. Running these models on-device
  would need exporting them to `frontend/public/models` and a JS decoder like
  `frontend/src/utils/potholeDetector.js`.
- No model has been trained and the CLIP weights have not been downloaded in the environment this was written in: the pipeline, scripts and wiring are tested with stubs (see `backend/tests/test_issue_*.py`), but the first real run of `train_clip_head.py` / `train_classifier.py` may need small fixes.
- `scripts/install_clip.py` does not pin a CLIP revision (it records the commit it downloaded in `models/CLIP_MANIFEST.json`); pin it before deploying.
