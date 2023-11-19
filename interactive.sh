#!/bin/bash

streamlit run interactive.py -- \
    --num_decoder_layers=6 \
    --use_color \
    --weight_decay=0.0005 \
    --data_root=data/ \
    --val_freq=3 \
    --batch_size=12 \
    --save_freq=3 \
    --print_freq=500 \
    --lr_backbone=2e-3 \
    --lr=2e-4 \
    --dataset=scanrefer \
    --test_dataset=scanrefer \
    --detect_intermediate \
    --joint_det \
    --use_soft_token_loss \
    --use_contrastive_align \
    --log_dir=data/output/logs/eda \
    --lr_decay_epochs 50 75 \
    --butd \
    --self_attend \
    --augment_det \
    --checkpoint_path=data/checkpoints/ScanRefer_54_59.pth \
    --eval
