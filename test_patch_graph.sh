#!/usr/bin/env bash
set -euo pipefail

# Patch-graph AA-CLIP batch testing script.
# Examples:
#   SAVE_PATH=./ckpt/aaclip_patch_graph bash test_patch_graph.sh
#   DATASETS="Brain Liver Retina" SAVE_PATH=./ckpt/aaclip_patch_graph bash test_patch_graph.sh

MODEL_NAME="${MODEL_NAME:-ViT-L-14-336}"
SAVE_PATH="${SAVE_PATH:-./ckpt/aaclip_patch_graph}"
IMG_SIZE="${IMG_SIZE:-518}"
SHOT="${SHOT:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-111}"

TEXT_ADAPT_WEIGHT="${TEXT_ADAPT_WEIGHT:-0.1}"
IMAGE_ADAPT_WEIGHT="${IMAGE_ADAPT_WEIGHT:-0.1}"
TEXT_ADAPT_UNTIL="${TEXT_ADAPT_UNTIL:-3}"
IMAGE_ADAPT_UNTIL="${IMAGE_ADAPT_UNTIL:-6}"

PATCH_GRAPH_K="${PATCH_GRAPH_K:-8}"
PATCH_GRAPH_ALPHA="${PATCH_GRAPH_ALPHA:-0.7}"
PATCH_GRAPH_RESIDUAL_WEIGHT="${PATCH_GRAPH_RESIDUAL_WEIGHT:-0.2}"

DATASETS="${DATASETS:-MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300}"

EXTRA_ARGS=()
if [[ "${DISABLE_PATCH_GRAPH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_patch_graph)
fi
if [[ "${DISABLE_PATCH_GRAPH_SPATIAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_patch_graph_spatial)
fi
if [[ "${RELU:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--relu)
fi
if [[ "${VISUALIZE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--visualize)
fi

echo "Testing patch-graph AA-CLIP"
echo "  save_path: $SAVE_PATH"
echo "  datasets:  $DATASETS"

for dataset in $DATASETS; do
  echo "========================================"
  echo "Testing $dataset"
  python test.py \
    --model_name "$MODEL_NAME" \
    --save_path "$SAVE_PATH" \
    --dataset "$dataset" \
    --shot "$SHOT" \
    --batch_size "$BATCH_SIZE" \
    --img_size "$IMG_SIZE" \
    --seed "$SEED" \
    --text_adapt_weight "$TEXT_ADAPT_WEIGHT" \
    --image_adapt_weight "$IMAGE_ADAPT_WEIGHT" \
    --text_adapt_until "$TEXT_ADAPT_UNTIL" \
    --image_adapt_until "$IMAGE_ADAPT_UNTIL" \
    --patch_graph_k "$PATCH_GRAPH_K" \
    --patch_graph_alpha "$PATCH_GRAPH_ALPHA" \
    --patch_graph_residual_weight "$PATCH_GRAPH_RESIDUAL_WEIGHT" \
    "${EXTRA_ARGS[@]}"
done
