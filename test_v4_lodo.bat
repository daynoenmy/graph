@echo off
cd /d "%~dp0"

set "TARGET=%~1"
if not defined TARGET set "TARGET=Chest"

python test_v4.py ^
  --dataset %TARGET% ^
  --save_path ./ckpt/v4_1_bmad_lodo/%TARGET% ^
  --checkpoint v4_1_head_latest.pth ^
  --prompt_source template ^
  --feature_layers 6 12 18 24 ^
  --laplacian_temperature 0.2 ^
  --spectral_uniform_mass 0.2 ^
  --max_spectral_coefficient 1.0 ^
  --readout_temperature 1.0 ^
  --test_noise_severity 0.0

if errorlevel 1 exit /b 1
