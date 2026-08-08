@echo off
cd /d "%~dp0"

python test_v4.py ^
  --dataset Liver ^
  --save_path ./ckpt/v4_2_ssc ^
  --checkpoint v4_2_ssc_head_epoch_*.pth ^
  --prompt_source template ^
  --feature_layers 6 12 18 24 ^
  --laplacian_temperature 0.2 ^
  --spectral_uniform_mass 0.2 ^
  --max_spectral_coefficient 1.0 ^
  --aspect_temperature 10.0 ^
  --aspect_coupling_strength 1.0 ^
  --readout_temperature 1.0 ^
  --test_noise_severity 0.0

if errorlevel 1 exit /b 1
