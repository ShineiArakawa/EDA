DATA_ROOT=data

TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 1111 \
    train_dist_mod.py \
    --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${DATA_ROOT}/ \
    --val_freq 3 \
    --batch_size 12 \
    --save_freq 3 \
    --print_freq 500 \
    --lr_backbone=2e-3 \
    --lr=2e-4 \
    --dataset scanrefer \
    --test_dataset scanrefer \
    --detect_intermediate \
    --joint_det \
    --use_soft_token_loss \
    --use_contrastive_align \
    --log_dir ${DATA_ROOT}/output/logs/eda \
    --lr_decay_epochs 50 75 \
    --butd \
    --self_attend \
    --augment_det \
    --checkpoint_path ${DATA_ROOT}/checkpoints/ScanRefer_54_59.pth \
    --eval
