@echo off
python test.py ^
  --dataset Brain ^
  --save_path ./ckpt/text_graph ^
  --patch_graph_k 8 ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2
