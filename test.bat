  python test.py ^
    --dataset Brain ^
    --maskless_datasets Chest HIS ^
    --save_path ckpt/cross_level_graph_1/leave_out_Brain ^
    --checkpoint image_adapter_3.pth ^
    --batch_size 16