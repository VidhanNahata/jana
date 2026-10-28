from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "JanaSutra API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    SECRET_KEY: str = "supersecretkey_janasutra_2026_dev_env_only"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DATABASE_URL: str = "sqlite:///./janasutra.db"
    POTHOLE_MODEL_PATH: str = "backend/ml_models/pothole.onnx"
    POTHOLE_CONF_THRESHOLD: float = 0.2
    # Dashcam road-damage model (pothole / cracks / faded markings); optional
    ROAD_DAMAGE_MODEL_PATH: str = "backend/ml_models/road_damage.onnx"
    ROAD_DAMAGE_CONFIG_PATH: str = "backend/ml_models/road_damage.json"
    # CLIP civic scene classifier (fallen tree, garbage, flooding, debris, open manhole);
    # int8 ONNX ~90 MB, roughly 150 MB more RAM, so it is off on 512 MB hosts
    CIVIC_SCENE_ENABLED: bool = True
    CIVIC_SCENE_MODEL_PATH: str = "backend/ml_models/civic_scene.onnx"
    CIVIC_SCENE_CONFIG_PATH: str = "backend/ml_models/civic_scene.json"
    # All-category issue models (garbage, drains, waterlogging, streetlights, trees, footpaths, signs...).
    # Optional: when the .onnx + .json files are absent they are skipped. See training/issues/README.md.
    ISSUE_MODELS_ENABLED: bool = True
    ISSUE_CLASSIFIER_PATH: str = "backend/ml_models/issue_classifier.onnx"
    ISSUE_DETECTOR_PATH: str = "backend/ml_models/issue_detector.onnx"
    # CLIP option: pretrained encoder (python scripts/install_clip.py) + a small trained head
    # (training/issues/train_clip_head.py --install). Both must exist; otherwise it is skipped.
    ISSUE_CLIP_MODEL_PATH: str = "models/clip-vit-base-patch32"
    ISSUE_CLIP_HEAD_PATH: str = "backend/ml_models/issue_clip_head.npz"
    ISSUE_MODEL_MAX_CONFIDENCE: float = 0.85   # a single model alone never claims more than this
    AUTHENTICITY_MODEL_PATH: str = "backend/ml_models/authenticity.onnx"
    AUTHENTICITY_THRESHOLD: float = 0.761  # from training/runs/authenticity/eval_report.json (<=3% false flags on real val photos)
    # ── Civic AI pipeline (backend/ai) ──────────────────────────────────
    # Strict offline: no network access for inference, no model downloads
    # (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE forced on, local_files_only).
    AI_STRICT_OFFLINE: bool = True
    # Local vision-language model directory (install with scripts/install_models.py).
    # Empty or missing -> image understanding is reported as model_unavailable
    # and the pipeline runs on text, the pothole detector and the registry.
    AI_VLM_ENABLED: bool = True
    AI_VLM_PATH: str = "models/qwen3-vl-2b-instruct"
    AI_VLM_DEVICE: str = "auto"          # auto | cuda | mps | cpu
    AI_VLM_DTYPE: str = "auto"           # auto | bfloat16 | float16 | float32
    AI_VLM_QUANTIZATION: str = "none"    # none | bnb-8bit | bnb-4bit (CUDA only) | dynamic-int8 (CPU only)
    AI_VLM_MAX_NEW_TOKENS: int = 384
    AI_VLM_MAX_IMAGE_SIDE: int = 640     # larger photos are downscaled before the VLM (measured setting)
    AI_VLM_TRANSLATE_KANNADA: bool = True  # use the local VLM's language model to translate Kannada text
    # Zero-shot text reading by the local VLM. Off: on the held-out set it raised
    # accuracy 0.52->0.60 but labelled 4 of 6 non-civic requests as potholes
    # (out-of-scope recall 0.50->0.17). See backend/ai/README.md.
    AI_LLM_TEXT_CLASSIFY: bool = False
    # Versioned knowledge files; empty = bundled files in backend/ai/data
    AI_TAXONOMY_PATH: str = ""
    AI_REGISTRY_PATH: str = ""
    AI_LEXICON_PATH: str = ""
    AI_GAZETTEER_PATH: str = ""
    # Below this classification confidence the complaint goes to human review
    AI_REVIEW_CONFIDENCE: float = 0.6
    # Comma-separated list of allowed browser origins, or "*"
    CORS_ORIGINS: str = "*"
    # Seeded account passwords — override via env in any public deployment
    ADMIN_PASSWORD: str = "admin123"
    DEMO_PASSWORD: str = "demo123"

    class Config:
        env_file = ".env"

settings = Settings()
