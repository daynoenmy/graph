@echo off
cd /d "%~dp0"

python train.py ^
  --dataset Brain ^
  --training_mode full_shot ^
  --save_path ./ckpt/noise_graph ^
  --patch_graph_k 8 ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2 ^
  --noise_severity 0.06 ^
  --noise_consistency_weight 0.1 ^
  --lesion_preservation_weight 0.1 ^
  --boundary_contrast_weight 0.05

if errorlevel 1 exit /b 1
