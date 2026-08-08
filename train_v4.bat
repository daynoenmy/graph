@echo off
cd /d "%~dp0"

python train_v4.py ^
  --dataset Brain ^
  --training_mode full_shot ^
  --save_path ./ckpt/v4_2_ssc ^
  --prompt_source template ^
  --feature_layers 6 12 18 24 ^
  --laplacian_temperature 0.2 ^
  --spectral_uniform_mass 0.2 ^
  --max_spectral_coefficient 1.0 ^
  --aspect_temperature 10.0 ^
  --aspect_coupling_strength 1.0 ^
  --readout_temperature 1.0 ^
  --image_loss_weight 1.0 ^
  --pixel_loss_weight 1.0 ^
  --epochs 10

if errorlevel 1 exit /b 1
