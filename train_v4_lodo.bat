@echo off
cd /d "%~dp0"

set "TARGET=%~1"
if not defined TARGET set "TARGET=Chest"

rem Shared defaults for non-Chest LODO folds.
set "LAPLACIAN_TEMPERATURE=0.2"
set "SPECTRAL_UNIFORM_MASS=0.2"
set "MAX_SPECTRAL_COEFFICIENT=1.0"
set "ASPECT_TEMPERATURE=10.0"
set "READOUT_TEMPERATURE=1.0"
set "IMAGE_LOSS_WEIGHT=1.0"
set "PIXEL_LOSS_WEIGHT=1.0"
set "EPOCHS=10"

rem Conservative Chest X-ray preset: suppress rib/texture high-frequency
rem over-correction and avoid saturated global aspect compatibilities.
if /I "%TARGET%"=="Chest" (
  set "LAPLACIAN_TEMPERATURE=0.15"
  set "SPECTRAL_UNIFORM_MASS=0.35"
  set "MAX_SPECTRAL_COEFFICIENT=0.75"
  set "ASPECT_TEMPERATURE=7.5"
  set "READOUT_TEMPERATURE=1.5"
  set "PIXEL_LOSS_WEIGHT=0.5"
)

python train_v4.py ^
  --lodo_target %TARGET% ^
  --training_mode full_shot ^
  --save_path ./ckpt/v4_2_ssc_bmad_lodo/%TARGET% ^
  --prompt_source template ^
  --feature_layers 6 12 18 24 ^
  --laplacian_temperature %LAPLACIAN_TEMPERATURE% ^
  --spectral_uniform_mass %SPECTRAL_UNIFORM_MASS% ^
  --max_spectral_coefficient %MAX_SPECTRAL_COEFFICIENT% ^
  --aspect_temperature %ASPECT_TEMPERATURE% ^
  --aspect_coupling_strength 1.0 ^
  --readout_temperature %READOUT_TEMPERATURE% ^
  --image_loss_weight %IMAGE_LOSS_WEIGHT% ^
  --pixel_loss_weight %PIXEL_LOSS_WEIGHT% ^
  --epochs %EPOCHS%

if errorlevel 1 exit /b 1
