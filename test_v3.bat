@echo off
cd /d "%~dp0"

rem Change Liver to another medical target dataset registered in constants.py.
python test_v3.py ^
  --dataset Liver ^
  --save_path ./ckpt/v3_multilayer_sfgraph ^
  --checkpoint v3_head_epoch_*.pth ^
  --prompt_source llm ^
  --llm_prompt_path ./dataset/llm_prompts.json ^
  --feature_layers 6 12 18 24 ^
  --hidden_dim 32 ^
  --text_temperature 10.0 ^
  --low_frequency_temperature 0.2 ^
  --high_frequency_temperature 1.0 ^
  --semantic_graph_temperature 0.1 ^
  --max_correction 4.0 ^
  --topk_ratio 0.05 ^
  --gem_power 3.0 ^
  --initial_cls_pool_weight 0.5 ^
  --initial_topk_pool_weight 0.5 ^
  --test_noise_severity 0.0

if errorlevel 1 exit /b 1
