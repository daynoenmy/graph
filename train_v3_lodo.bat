@echo off
cd /d "%~dp0"

set "TARGET=%~1"
if not defined TARGET set "TARGET=Brain"

python train_v3.py ^
  --lodo_target %TARGET% ^
  --training_mode full_shot ^
  --save_path ./ckpt/v3_bmad_lodo/%TARGET% ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/bmad_llm_prompts.json ^
  --feature_layer 18 ^
  --hidden_dim 32 ^
  --semantic_graph_temperature 0.1 ^
  --max_correction 4.0 ^
  --band_consistency_weight 0.05 ^
  --lesion_preservation_weight 0.05 ^
  --band_scale_min 0.5 ^
  --band_scale_max 1.5 ^
  --epochs 10

if errorlevel 1 exit /b 1
