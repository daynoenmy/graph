@echo off
cd /d "%~dp0"

set "TARGET=%~1"
if not defined TARGET set "TARGET=Brain"

python test_v3.py ^
  --dataset %TARGET% ^
  --save_path ./ckpt/v3_bmad_lodo_multilayer/%TARGET% ^
  --checkpoint v3_head_latest.pth ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/bmad_llm_prompts.json ^
  --feature_layers 6 12 18 24 ^
  --hidden_dim 32 ^
  --semantic_graph_temperature 0.1 ^
  --max_correction 4.0 ^
  --topk_ratio 0.05 ^
  --gem_power 3.0 ^
  --initial_cls_pool_weight 0.5 ^
  --initial_topk_pool_weight 0.5 ^
  --test_noise_severity 0.0

if errorlevel 1 exit /b 1
