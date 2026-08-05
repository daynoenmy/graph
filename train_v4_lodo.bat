@echo off
cd /d "%~dp0"

set "TARGET=%~1"
if not defined TARGET set "TARGET=Chest"

python train_v4.py ^
  --lodo_target %TARGET% ^
  --training_mode full_shot ^
  --save_path ./ckpt/v4_bmad_lodo/%TARGET% ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/bmad_llm_prompts.json ^
  --feature_layers 6 12 18 24 ^
  --laplacian_temperature 0.2 ^
  --spectral_uniform_mass 0.2 ^
  --readout_temperature 1.0 ^
  --epochs 10

if errorlevel 1 exit /b 1
