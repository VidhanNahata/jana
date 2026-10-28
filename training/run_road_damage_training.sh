#!/bin/sh
# Train the 4-class dashcam road-damage model (data from prepare_road_damage.py)
# one epoch per process: epoch 1 starts from the trained pothole detector's
# backbone, then supervise_training.sh resumes until all epochs are done.
# Usage: EPOCHS=10 sh training/run_road_damage_training.sh
cd "$(dirname "$0")" || exit 1
RUN=runs/road_damage_yolo11n
LOG=runs/road_damage_training.log
step() { echo "[road] $* ($(date '+%H:%M'))" | tee -a $LOG; }

if [ ! -f $RUN/weights/last.pt ]; then
  step "training epoch 1 (from the pothole detector backbone)"
  caffeinate -i python3 train.py --model yolo11n.yaml --init runs/pothole_yolo11n/weights/deployed_epoch11.pt \
    --data data/road_damage/data.yaml --name road_damage_yolo11n --epochs ${EPOCHS:-10} --batch 6 --one-epoch >> $LOG 2>&1
  [ -f $RUN/weights/last.pt ] || { step "first epoch FAILED"; exit 1; }
fi
step "resuming epoch by epoch"
sh supervise_training.sh $RUN/weights/last.pt 6 >> $LOG 2>&1 || { step "training FAILED"; exit 1; }
step "all done"
