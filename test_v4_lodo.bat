@echo off
cd /d "%~dp0"

set "TARGET=%~1"
if not defined TARGET set "TARGET=Chest"

rem These architecture values must match the corresponding training fold.
set "LAPLACIAN_TEMPERATURE=0.2"
set "SPECTRAL_UNIFORM_MASS=0.2"
set "MAX_SPECTRAL_COEFFICIENT=1.0"
set "ASPECT_TEMPERATURE=10.0"
set "READOUT_TEMPERATURE=1.0"

if /I "%TARGET%"=="Chest" (
  set "LAPLACIAN_TEMPERATURE=0.15"
  set "SPECTRAL_UNIFORM_MASS=0.35"
  set "MAX_SPECTRAL_COEFFICIENT=0.75"
  set "ASPECT_TEMPERATURE=7.5"
  set "READOUT_TEMPERATURE=1.5"
)

python test_v4.py ^
  --dataset %TARGET% ^
  --save_path ./ckpt/v4_2_ssc_bmad_lodo/%TARGET% ^
  --checkpoint v4_2_ssc_head_latest.pth ^
  --prompt_source template ^
  --feature_layers 6 12 18 24 ^
  --laplacian_temperature %LAPLACIAN_TEMPERATURE% ^
  --spectral_uniform_mass %SPECTRAL_UNIFORM_MASS% ^
  --max_spectral_coefficient %MAX_SPECTRAL_COEFFICIENT% ^
  --aspect_temperature %ASPECT_TEMPERATURE% ^
  --aspect_coupling_strength 1.0 ^
  --readout_temperature %READOUT_TEMPERATURE% ^
  --test_noise_severity 0.0

if errorlevel 1 exit /b 1
