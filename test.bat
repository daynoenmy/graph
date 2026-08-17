  python test.py ^
    --dataset Brain ^
    --maskless_datasets Chest HIS ^
    --save_path ckpt/cross_level_graph_2/leave_out_Brain ^
    --checkpoint image_adapter_1.pth ^
    --batch_size 16 ^
    --seed 1