@echo off
cd /d "%~dp0"

python train.py ^
  --dataset Brain ^
  --training_mode full_shot ^
  --save_path ./ckpt/noise_graph_cls_llm ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/llm_prompts.json ^
  --patch_graph_k 8 ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2 ^
  --clip_global_weight 0.2 ^
  --global_text_temperature 10.0 ^
  --noise_severity 0.06 ^
  --noise_consistency_weight 0.1 ^
  --lesion_preservation_weight 0.1 ^
  --boundary_contrast_weight 0.05

if errorlevel 1 exit /b 1
