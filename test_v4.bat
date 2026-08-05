@echo off
cd /d "%~dp0"

python test_v4.py ^
  --dataset Liver ^
  --save_path ./ckpt/v4_graph_spectral ^
  --checkpoint v4_head_epoch_*.pth ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/bmad_llm_prompts.json ^
  --feature_layers 6 12 18 24 ^
  --laplacian_temperature 0.2 ^
  --spectral_uniform_mass 0.2 ^
  --readout_temperature 1.0 ^
  --test_noise_severity 0.0

if errorlevel 1 exit /b 1
