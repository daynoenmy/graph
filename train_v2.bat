@echo off
cd /d "%~dp0"

python train.py ^
  --dataset Brain ^
  --training_mode full_shot ^
  --save_path ./ckpt/noise_graph_v2 ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2 ^
  --patch_graph_soft ^
  --patch_graph_spectral_norm ^
  --graph_primary_only ^
  --train_noise_types additive magnitude signal_dependent multiplicative low_frequency ^
  --primary_noise_probability 0.7 ^
  --noise_severity_min 0.0 ^
  --noise_severity_max 0.10 ^
  --num_noise_views 2 ^
  --noise_consistency_weight 0.1 ^
  --noise_balance_weight 0.05 ^
  --lesion_preservation_weight 0.1 ^
  --boundary_contrast_weight 0.05 ^
  --min_lesion_contrast_retention 0.7

if errorlevel 1 exit /b 1
