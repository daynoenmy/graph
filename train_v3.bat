@echo off
cd /d "%~dp0"

python train_v3.py ^
  --dataset Brain ^
  --training_mode full_shot ^
  --save_path ./ckpt/v3_frozen_sfgraph ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/llm_prompts.json ^
  --feature_layer 18 ^
  --hidden_dim 32 ^
  --text_temperature 10.0 ^
  --low_frequency_temperature 0.2 ^
  --high_frequency_temperature 1.0 ^
  --image_pool_temperature 10.0 ^
  --band_consistency_weight 0.05 ^
  --epochs 10

if errorlevel 1 exit /b 1
