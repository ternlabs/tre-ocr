#!/usr/bin/env bash
# Fetch the round 1 recogniser that tools/read_pdf.py reads with: the weights set F was
# measured with (METHOD.md), from the Weights url in pyproject.toml (their own repository).
# The archive's hash is checked here; read_pdf.py checks the weights' own hash again.
# Usage, from the repository root:  bash tools/get_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ARCHIVE_SHA256=010c5757be568a258d6657402a3acbbf6450f46dd978840e4f5a298e1e22fd5e
DEST=cache/train/output            # read_pdf.py's ROUND1 is $DEST/inference
URL=$(sed -n 's/^Weights *= *"\(.*\)"/\1/p' pyproject.toml)
[ -n "$URL" ] || { echo "no Weights url in pyproject.toml" >&2; exit 1; }

mkdir -p cache "$DEST"
tmp=$(mktemp cache/weights.XXXXXX); trap 'rm -f "$tmp"' EXIT
echo "fetching $URL"
curl -fL --retry 3 -o "$tmp" "$URL"
got=$(sha256sum "$tmp" | cut -c1-64)
[ "$got" = "$ARCHIVE_SHA256" ] || { echo "archive hash $got is not the published one: not unpacked" >&2; exit 1; }
rm -rf "$DEST/inference"
tar -xzf "$tmp" -C "$DEST" --no-same-owner
echo "weights in $DEST/inference"
