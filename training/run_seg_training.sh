#!/bin/sh
# Wait for prepare_seg_dataset.py, then train the YOLO11n-seg pothole model one
# epoch per process (first epoch fresh from the trained detector, then resume),
# then evaluate on the detection test set and export the ONNX for the app.
# Usage: sh training/run_seg_training.sh
cd "$(dirname "$0")" || exit 1
RUN=runs/pothole_yolo11n_seg
LOG=runs/seg_training.log
step() { echo "[seg] $* ($(date '+%H:%M'))" | tee -a $LOG; }

# Train/val image lists (only images that have polygon labels) are built by the caller:
# data/potholes_seg/{train,val}_labelled.txt and data/potholes_seg/data_labelled.yaml
if [ ! -f $RUN/weights/last.pt ]; then
  step "training epoch 1 (from the epoch-11 detector)"
  caffeinate -i python3 train.py --model yolo11n-seg.yaml --init runs/pothole_yolo11n/weights/deployed_epoch11.pt \
    --data data/potholes_seg/data_labelled.yaml --name pothole_yolo11n_seg --epochs ${EPOCHS:-5} --batch 6 --one-epoch >> $LOG 2>&1
  [ -f $RUN/weights/last.pt ] || { step "first epoch FAILED"; exit 1; }
fi
step "resuming epoch by epoch"
sh supervise_training.sh $RUN/weights/last.pt 6 >> $LOG 2>&1 || { step "training FAILED"; exit 1; }
step "evaluating"
python3 evaluate.py $RUN/weights/best.pt >> $LOG 2>&1 || { step "evaluation FAILED"; exit 1; }
step "all done"
