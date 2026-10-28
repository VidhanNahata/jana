#!/bin/sh
# Overnight chain: when the pothole run finishes, evaluate + export it, then
# build and train the photo authenticity classifier. Deployment is done by hand
# after reviewing the reports.
# Usage: sh training/run_after_pothole.sh
cd "$(dirname "$0")/.." || exit 1
RUN=training/runs/pothole_yolo11n
LOG=training/runs/after_pothole.log
step() { echo "[chain] $* ($(date '+%H:%M'))" | tee -a $LOG; }

until [ -f $RUN/weights/DONE ]; do
  pgrep -f supervise_training.sh >/dev/null || { step "pothole supervisor not running and no DONE marker; stopping"; exit 1; }
  sleep 60
done

step "pothole training done; evaluating"
python3 training/evaluate.py $RUN/weights/best.pt >> $LOG 2>&1 || step "pothole evaluation FAILED"
python3 training/export.py $RUN/weights/best.pt >> $LOG 2>&1 || step "pothole export FAILED"

step "generating AI road images"
if python3 -c "import diffusers" 2>/dev/null; then
  SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())") env -u HF_TOKEN \
    python3 training/authenticity/generate_ai_roads.py >> $LOG 2>&1 || step "generation FAILED (continuing without it)"
else
  step "diffusers not installed; skipping local AI road images"
fi

step "building authenticity dataset"
python3 training/authenticity/prepare_authenticity.py >> $LOG 2>&1 || { step "dataset build FAILED"; exit 1; }
step "training authenticity classifier"
python3 training/authenticity/train_authenticity.py --epochs 8 --workers 3 >> $LOG 2>&1 || { step "authenticity training FAILED"; exit 1; }
step "evaluating + exporting authenticity classifier"
python3 training/authenticity/evaluate_export.py >> $LOG 2>&1 || { step "authenticity eval/export FAILED"; exit 1; }
step "all done"
