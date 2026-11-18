#!/usr/bin/env bash
# Step 8 on the GPU machine: Qwen2.5-VL 7B over sets C, A and B.
#
# Run from a clone of the repo, with the page images copied into cache/set-a,
# cache/set-b and cache/set-c. Writes data/runs/qwen7b-<weights>-<set>.jsonl and their
# .meta.json; copy those back, nothing else. Stops and resumes: run it again.
#
# 8-bit weights on a card of 11.5 GB or more, else Qwen's AWQ 4-bit release;
# WEIGHTS=8bit or WEIGHTS=awq overrides. SETS="c" runs fewer sets.
set -euo pipefail
cd "$(dirname "$0")/.."

for s in ${SETS:-c a b}; do
  compgen -G "cache/set-$s/*.png" > /dev/null || { echo "cache/set-$s holds no page images"; exit 1; }
done

if ! .venv-lab/bin/python -c "import torch, transformers" 2>/dev/null; then
  python3 -m venv .venv-lab
  .venv-lab/bin/pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  .venv-lab/bin/pip install -q -r requirements-lab.txt
fi

# FlashAttention-2 from its prebuilt wheel: building it needs the CUDA toolkit (nvcc).
if ! .venv-lab/bin/python -c "import flash_attn" 2>/dev/null; then
  py=$(.venv-lab/bin/python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
  abi=$(.venv-lab/bin/python -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')
  .venv-lab/bin/pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${abi}-${py}-${py}-linux_x86_64.whl" \
    || true
fi
if ! err=$(.venv-lab/bin/python -c "import flash_attn" 2>&1); then
  echo "flash-attn does not load, and without it the image encoder does not fit:"
  echo "$err" | tail -3
  exit 1
fi

mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
weights=${WEIGHTS:-$([ "$mib" -ge 11500 ] && echo 8bit || echo awq)}
if [ "$weights" = awq ]; then
  .venv-lab/bin/pip install -q autoawq==0.2.8
fi
echo "GPU memory ${mib} MiB: ${weights} weights"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
need=$([ "$weights" = 8bit ] && echo 10000 || echo 7000)
if [ "$free" -lt "$need" ]; then
  echo "only ${free} MiB free of ${mib}, ${need} needed; the GPU is held by:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # page images differ in size
for s in ${SETS:-c a b}; do
  .venv-lab/bin/python -I tools/run_engine.py sets/disclosures-$s.jsonl --images cache/set-$s \
    --engine qwen --weights "$weights" --out data/runs/qwen7b-$weights-$s.jsonl --machine lab
done
