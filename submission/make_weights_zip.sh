#!/usr/bin/env bash
# Package a weights directory (laid out as in the README) as the weights.zip the platform mounts at /weights.
set -euo pipefail
WEIGHTS="${1:?usage: make_weights_zip.sh <weights-dir> [weights.zip]}"
OUT="$(realpath -m "${2:-weights.zip}")"
FILES=(adapter.pt native_spacing_table.json inplane_spacing_table.json diff_unet_3d_rflow-mr-brain_v0.pt
       autoencoder_v1.pt MedEmbed-large-v0.1 Bio_ClinicalBERT BiomedVLP-CXR-BERT-specialized)
for f in "${FILES[@]}"; do
    [[ -e "$WEIGHTS/$f" ]] || { echo "missing $WEIGHTS/$f (see 'Pretrained Checkpoint' in the README)" >&2; exit 1; }
done
rm -f "$OUT"
cd "$WEIGHTS" && zip -r0 -q "$OUT" "${FILES[@]}" -x '*/.cache/*'
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
