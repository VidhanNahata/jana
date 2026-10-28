#!/bin/sh
# Resume a YOLO run one epoch per process until it finishes (see train.py --one-epoch).
# Usage: sh training/supervise_training.sh training/runs/<name>/weights/last.pt [batch]
LAST=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
BATCH=${2:-6}
cd "$(dirname "$0")/weights" || exit 1
rm -f "$(dirname "$LAST")/DONE"
fails=0
while [ ! -f "$(dirname "$LAST")/DONE" ]; do
  caffeinate -i python3 ../train.py --resume "$LAST" --batch "$BATCH" --one-epoch
  code=$?
  echo "[supervisor] process exited with $code at $(date)"
  if [ $code -eq 75 ] || [ $code -eq 0 ]; then fails=0; else fails=$((fails+1)); fi
  [ $fails -ge 3 ] && { echo "[supervisor] 3 consecutive failures, giving up"; exit 1; }
  sleep 10
done
echo "[supervisor] training complete"
