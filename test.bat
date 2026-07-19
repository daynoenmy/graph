@echo off
cd /d "%~dp0"

rem Change Liver to any target dataset registered in dataset\constants.py.
python test.py ^
  --dataset Liver ^
  --save_path ./ckpt/noise_graph ^
  --patch_graph_k 8 ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2 ^
  --noise_severity 0.06

if errorlevel 1 exit /b 1
