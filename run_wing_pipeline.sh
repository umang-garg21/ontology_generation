#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_wing_pipeline.sh \
    --image /path/to/image.jpg \
    [--cropped /path/to/cropped_rotated.png] \
    [--masks out.ome.tiff] \
    [--device cpu|cuda|cuda:0] \
    [--tokens /path/to/mask_tokens.npy] \
    [--png-dir /path/to/mask_pngs] \
    [--overlap 0.7] \
    [--threshold 0.5] \
    [--prompt "wing"]

Required:
  --image      Input image path

Optional:
  --cropped    Output cropped (rotated) image path (default: data/cropped_rotated/<stem>_cropped_rotated.png)
  --flip-vertical  Flip the image vertically after rotation
  --flip-horizontal  Flip the image horizontally after rotation
  --masks      Output OME-TIFF filename (default: out.ome.tiff)

Optional:
  --device     cpu|cuda|cuda:0 (default: auto)
  --tokens     Output .npy for per-mask tokens (default: alongside TIFF)
  --png-dir    Output folder for per-mask PNGs (default: <tiff_dir>/mask_pngs)
  --points-per-batch  Points per batch for SAM mask generation (default: unset)
  --overlap    Overlap threshold (default: 0.7)
  --threshold  SAM3 detection threshold (default: 0.5)
  --prompt     SAM3 text prompt (default: wing)
EOF
}

IMAGE=""
CROPPED=""
MASKS=""
DEVICE="cpu"
FLIP_VERTICAL="false"
FLIP_HORIZONTAL="false"
TOKENS=""
PNG_DIR=""
POINTS_PER_BATCH=""
OVERLAP="0.7"
THRESHOLD="0.5"
PROMPT="wing"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"; shift 2;;
    --cropped)
      CROPPED="$2"; shift 2;;
    --flip-vertical)
      FLIP_VERTICAL="true"; shift;;
    --flip-horizontal)
      FLIP_HORIZONTAL="true"; shift;;
    --masks)
      MASKS="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --tokens)
      TOKENS="$2"; shift 2;;
    --png-dir)
      PNG_DIR="$2"; shift 2;;
    --points-per-batch)
      POINTS_PER_BATCH="$2"; shift 2;;
    --overlap)
      OVERLAP="$2"; shift 2;;
    --threshold)
      THRESHOLD="$2"; shift 2;;
    --prompt)
      PROMPT="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  usage
  exit 1
fi

IMAGE_BASENAME="$(basename "$IMAGE")"
IMAGE_STEM="${IMAGE_BASENAME%.*}"
OUTPUT_DIR="$(pwd)/output_masks/${IMAGE_STEM}"
mkdir -p "$OUTPUT_DIR"

if [[ -z "$CROPPED" ]]; then
  CROPPED_DIR="$(pwd)/data/cropped_rotated"
  mkdir -p "$CROPPED_DIR"
  CROPPED_SUFFIX=""
  if [[ "$FLIP_VERTICAL" == "true" ]]; then
    CROPPED_SUFFIX+="_flipped_v"
  fi
  if [[ "$FLIP_HORIZONTAL" == "true" ]]; then
    CROPPED_SUFFIX+="_flipped_h"
  fi
  CROPPED="$CROPPED_DIR/${IMAGE_STEM}_cropped_rotated${CROPPED_SUFFIX}.png"
fi

if [[ -z "$MASKS" ]]; then
  MASKS="out.ome.tiff"
fi
MASKS_FILENAME="$(basename "$MASKS")"
MASKS="$OUTPUT_DIR/$MASKS_FILENAME"

if [[ -z "$TOKENS" ]]; then
  TOKENS="${MASKS%.*}_mask_tokens.npy"
fi

if [[ -z "$PNG_DIR" ]]; then
  PNG_DIR="$OUTPUT_DIR/mask_pngs"
fi

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"

IMAGE="$IMAGE" MASKS="$MASKS" DEVICE="$DEVICE" TOKENS="$TOKENS" PNG_DIR="$PNG_DIR" \
POINTS_PER_BATCH="$POINTS_PER_BATCH" OVERLAP="$OVERLAP" THRESHOLD="$THRESHOLD" PROMPT="$PROMPT" CROPPED="$CROPPED" FLIP_VERTICAL="$FLIP_VERTICAL" FLIP_HORIZONTAL="$FLIP_HORIZONTAL" \
"$PYTHON_BIN" - <<'PY'
import os
from segment_wing_augmented import run_pipeline

image = os.environ["IMAGE"]
masks = os.environ["MASKS"]
device = os.environ.get("DEVICE") or None
tokens = os.environ.get("TOKENS") or None
png_dir = os.environ.get("PNG_DIR") or None
cropped = os.environ.get("CROPPED") or None
flip_vertical = os.environ.get("FLIP_VERTICAL", "false").lower() == "true"
flip_horizontal = os.environ.get("FLIP_HORIZONTAL", "false").lower() == "true"
ppb = os.environ.get("POINTS_PER_BATCH") or None
overlap = float(os.environ.get("OVERLAP", "0.7"))
threshold = float(os.environ.get("THRESHOLD", "0.5"))
prompt = os.environ.get("PROMPT", "wing")

result = run_pipeline(
  image_path=image,
  output_cropped_path=cropped,
  output_masks_path=masks,
  output_tokens_path=tokens,
  output_masks_png_dir=png_dir,
  text_prompt=prompt,
  threshold=threshold,
  overlap_threshold=overlap,
  points_per_batch=int(ppb) if ppb else None,
  visualize=False,
  device=device,
  flip_vertical=flip_vertical,
  flip_horizontal=flip_horizontal,
)

print("cropped_path:", result.get("cropped_path"))
print("masks_path:", result.get("masks_path"))
print("tokens_path:", result.get("tokens_path"))
print("mask_png_paths:", result.get("mask_png_paths"))
PY
