@echo off
python train.py ^
  --dataset Brain ^
  --training_mode full_shot ^
  --save_path ./ckpt/text_graph ^
  --patch_graph_k 8 ^
  --patch_graph_alpha 0.7 ^
  --patch_graph_residual_weight 0.2
