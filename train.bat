 python train.py ^
    --leave_out Liver ^
    --datasets Chest Liver Brain OCT2017 RESC HIS ^
    --training_mode full_shot ^
    --maskless_datasets Chest HIS  OCT2017 ^
    --save_path ckpt/cross_level_graph_2/leave_out_Liver