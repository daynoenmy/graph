#!/usr/bin/env bash
set -euo pipefail

# Patch-graph AA-CLIP training script.
# Override any variable from the command line, for example:
#   DATASET=Brain SAVE_PATH=./ckpt/brain_patch_graph bash run_patch_graph.sh

MODEL_NAME="${MODEL_NAME:-ViT-L-14-336}"
DATASET="${DATASET:-VisA}"
TRAINING_MODE="${TRAINING_MODE:-full_shot}"
SHOT="${SHOT:-32}"
SAVE_PATH="${SAVE_PATH:-./ckpt/aaclip_patch_graph}"
IMG_SIZE="${IMG_SIZE:-518}"
SURGERY_UNTIL_LAYER="${SURGERY_UNTIL_LAYER:-20}"
SEED="${SEED:-111}"

TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-16}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-2}"
TEXT_EPOCH="${TEXT_EPOCH:-5}"
IMAGE_EPOCH="${IMAGE_EPOCH:-20}"
TEXT_LR="${TEXT_LR:-0.00001}"
IMAGE_LR="${IMAGE_LR:-0.0005}"

TEXT_ADAPT_WEIGHT="${TEXT_ADAPT_WEIGHT:-0.1}"
IMAGE_ADAPT_WEIGHT="${IMAGE_ADAPT_WEIGHT:-0.1}"
TEXT_ADAPT_UNTIL="${TEXT_ADAPT_UNTIL:-3}"
IMAGE_ADAPT_UNTIL="${IMAGE_ADAPT_UNTIL:-6}"
TEXT_NORM_WEIGHT="${TEXT_NORM_WEIGHT:-0.1}"

PATCH_GRAPH_K="${PATCH_GRAPH_K:-8}"
PATCH_GRAPH_ALPHA="${PATCH_GRAPH_ALPHA:-0.7}"
PATCH_GRAPH_RESIDUAL_WEIGHT="${PATCH_GRAPH_RESIDUAL_WEIGHT:-0.2}"

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

mkdir -p "$SAVE_PATH"

echo "Training patch-graph AA-CLIP"
echo "  dataset:       $DATASET"
echo "  mode:          $TRAINING_MODE"
echo "  save_path:     $SAVE_PATH"
echo "  graph k/alpha: $PATCH_GRAPH_K / $PATCH_GRAPH_ALPHA"

python train.py \
  --model_name "$MODEL_NAME" \
  --dataset "$DATASET" \
  --training_mode "$TRAINING_MODE" \
  --shot "$SHOT" \
  --save_path "$SAVE_PATH" \
  --img_size "$IMG_SIZE" \
  --surgery_until_layer "$SURGERY_UNTIL_LAYER" \
  --seed "$SEED" \
  --text_batch_size "$TEXT_BATCH_SIZE" \
  --image_batch_size "$IMAGE_BATCH_SIZE" \
  --text_epoch "$TEXT_EPOCH" \
  --image_epoch "$IMAGE_EPOCH" \
  --text_lr "$TEXT_LR" \
  --image_lr "$IMAGE_LR" \
  --text_norm_weight "$TEXT_NORM_WEIGHT" \
  --text_adapt_weight "$TEXT_ADAPT_WEIGHT" \
  --image_adapt_weight "$IMAGE_ADAPT_WEIGHT" \
  --text_adapt_until "$TEXT_ADAPT_UNTIL" \
  --image_adapt_until "$IMAGE_ADAPT_UNTIL" \
  --patch_graph_k "$PATCH_GRAPH_K" \
  --patch_graph_alpha "$PATCH_GRAPH_ALPHA" \
  --patch_graph_residual_weight "$PATCH_GRAPH_RESIDUAL_WEIGHT" \
  "${EXTRA_ARGS[@]}"
