@echo off
cd /d "%~dp0"

rem The target dataset selects its medically corresponding test corruption.
python test.py ^
  --dataset Liver ^
  --save_path ./ckpt/noise_graph_v2 ^
  --image_checkpoint "image_adapter_*.pth" ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2 ^
  --patch_graph_soft ^
  --patch_graph_spectral_norm ^
  --graph_primary_only ^
  --noise_severity 0.06 ^
  --probe_noise_type auto ^
  --test_noise_type auto ^
  --test_noise_severity 0.06

if errorlevel 1 exit /b 1
