#!/usr/bin/env bash
# Step 8 on the GPU machine: PaddleOCR-VL 1.6 and SenOCR-Vi over sets C, A and B.
#
# Run from a clone of the repo, with the page images copied into cache/set-a,
# cache/set-b and cache/set-c. Writes data/runs/paddleocr-vl-<set>.jsonl and
# data/runs/senocr-vi-<set>.jsonl with their .meta.json; copy those back, nothing else.
# Stops and resumes: run it again. SETS="c" runs fewer sets; MODELS="senocr-vi" fewer models.
#
# Needs a card with bfloat16: in float32 the weights do not fit 8 GB, and Paddle's
# bfloat16 needs compute capability 8.0 (Ampere) or later.
set -euo pipefail
cd "$(dirname "$0")/.."

for s in ${SETS:-c a b}; do
  compgen -G "cache/set-$s/*.png" > /dev/null || { echo "cache/set-$s holds no page images"; exit 1; }
done

if ! .venv-paddle/bin/python -c "import paddleocr, paddle" 2>/dev/null; then
  python3 -m venv .venv-paddle
  .venv-paddle/bin/pip install -q --default-timeout=120 --retries 5 \
    paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  .venv-paddle/bin/pip install -q --default-timeout=120 --retries 5 "paddleocr[doc-parser]==3.7.0" paddlex==3.7.2
fi

if ! .venv-paddle/bin/python -c "import paddle, sys; sys.exit(not paddle.amp.is_bfloat16_supported())" 2>/dev/null; then
  echo "this card has no bfloat16 in Paddle, and float32 does not fit"
  exit 1
fi
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 8000 ]; then
  echo "only ${free} MiB free, 8000 needed; the GPU is held by:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 1
fi

for m in ${MODELS:-paddleocr-vl senocr-vi}; do
  case $m in
    paddleocr-vl) id=PaddlePaddle/PaddleOCR-VL-1.6 ;;
    senocr-vi)    id=VietAlphaLabs/SenOCR-Vi ;;
    *) echo "unknown model $m"; exit 1 ;;
  esac
  for s in ${SETS:-c a b}; do
    .venv-paddle/bin/python -I tools/run_engine.py sets/disclosures-$s.jsonl --images cache/set-$s \
      --engine paddlevl --model "$id" --out data/runs/$m-$s.jsonl --machine lab
  done
done
