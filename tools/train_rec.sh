#!/usr/bin/env bash
# Fine-tune PaddleOCR's recogniser on tools/train_lines.py's lines, then export it.
# Runs the PaddleOCR repository's training code at a pinned commit, kept under the
# gitignored cache/train. Stopped, run it again: it resumes from the last epoch saved.
#
# Usage, from the repo root:  bash tools/train_rec.sh
set -euo pipefail

COMMIT=dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf
WEIGHTS=PP-OCRv6_medium_rec_pretrained.pdparams
WEIGHTS_URL=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/$WEIGHTS
WEIGHTS_SHA=c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92

ROOT=$(pwd)
PY=$ROOT/.venv-paddle/bin/python
TRAIN=$ROOT/cache/train
CONFIG=$ROOT/tools/train_rec.yml

[ -s "$TRAIN/data/train_list.txt" ] || { echo "no lines yet: run tools/train_lines.py" >&2; exit 1; }

if [ ! -d "$TRAIN/PaddleOCR/.git" ]; then
  git init -q "$TRAIN/PaddleOCR"
  git -C "$TRAIN/PaddleOCR" remote add origin https://github.com/PaddlePaddle/PaddleOCR.git
  git -C "$TRAIN/PaddleOCR" fetch -q --depth 1 origin "$COMMIT"
  git -C "$TRAIN/PaddleOCR" checkout -q FETCH_HEAD
fi
[ "$(git -C "$TRAIN/PaddleOCR" rev-parse HEAD)" = "$COMMIT" ] || { echo "PaddleOCR is not at $COMMIT" >&2; exit 1; }

[ -f "$TRAIN/$WEIGHTS" ] || curl -sS -o "$TRAIN/$WEIGHTS" "$WEIGHTS_URL"
echo "$WEIGHTS_SHA  $TRAIN/$WEIGHTS" | sha256sum -c --quiet -

"$PY" -m pip install -q lmdb albumentations albucore rapidfuzz shapely scikit-image pyclipper

cd "$TRAIN/PaddleOCR"
mkdir -p ../output
"$PY" -m pip freeze > ../output/packages.txt

RESUME=()
[ -f ../output/latest.pdparams ] && RESUME=(-o Global.checkpoints=../output/latest)
"$PY" tools/train.py -c "$CONFIG" "${RESUME[@]}" 2>&1 | tee -a ../output/train.log

"$PY" tools/export_model.py -c "$CONFIG" \
  -o Global.pretrained_model=../output/best_accuracy Global.save_inference_dir=../output/inference

echo "done: $TRAIN/output/inference"
