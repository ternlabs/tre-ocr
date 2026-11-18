#!/usr/bin/env bash
# Round 2: go on from round 1's recogniser with the hand-labelled lines mixed in
# (tools/hand_lines.py), then export the last epoch. Round 1's output stays as it is.
# The epochs and rate are fixed here, before any result: the typed validation lines do
# not hold the case round 2 is for, so they do not pick the epoch.
#
# Usage, from the repo root, after tools/train_rec.sh:  bash tools/train_round2.sh
set -euo pipefail

ROOT=$(pwd)
PY=$ROOT/.venv-paddle/bin/python
TRAIN=$ROOT/cache/train
CONFIG=$ROOT/tools/train_rec.yml
SET=(Global.save_model_dir=../output2 Global.pretrained_model=../output/best_accuracy
     Global.epoch_num=6 Optimizer.lr.learning_rate=0.0001
     "Train.dataset.label_file_list=['../data/round2_list.txt']")

[ -s "$TRAIN/data/round2_list.txt" ] || { echo "no mix yet: run tools/hand_lines.py mix" >&2; exit 1; }

cd "$TRAIN/PaddleOCR"
mkdir -p ../output2
"$PY" tools/train.py -c "$CONFIG" -o "${SET[@]}" 2>&1 | tee -a ../output2/train.log

"$PY" tools/export_model.py -c "$CONFIG" \
  -o Global.pretrained_model=../output2/latest Global.save_inference_dir=../output2/inference

echo "done: $TRAIN/output2/inference"
