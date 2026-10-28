# JanaSutra API — Render free web service (512 MB RAM, CPU)
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
# AI_VLM_ENABLED=false: 512 MB cannot hold the local VLM, so the civic AI
# pipeline runs the lexicon + ONNX detectors only (see backend/ai/README.md)
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    AI_VLM_ENABLED=false \
    AI_STRICT_OFFLINE=true
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Includes backend/ml_models/pothole_yolov8s.onnx
COPY --chown=user backend ./backend

# Render injects $PORT (default 10000)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
