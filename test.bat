@echo off
cd /d "%~dp0"

rem Change Liver to any target dataset registered in dataset\constants.py.
python test.py ^
  --dataset Liver ^
  --save_path ./ckpt/noise_graph_cls_llm ^
  --image_checkpoint image_adapter_*.pth ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/llm_prompts.json ^
  --patch_graph_k 8 ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2 ^
  --noise_severity 0.06 ^
  --clip_global_weight 0.2 ^
  --global_text_temperature 10.0 ^
  --medical_image_score_global_weight 0.2

if errorlevel 1 exit /b 1
