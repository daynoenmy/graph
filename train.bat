 python train.py ^
    --leave_out Brain ^
    --datasets Chest Liver Brain OCT2017 RESC HIS ^
    --training_mode full_shot ^
    --maskless_datasets Chest HIS  OCT2017 ^
    --save_path ckpt/cross_level_graph_1/leave_out_Brain