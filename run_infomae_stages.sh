#!/bin/bash

# InfoMAE 4-Stage Training Script
# 각 단계별로 다른 설정으로 훈련을 진행합니다.

set -e

# 기본 설정
DATA_PATH="./data/imagenet100"
OUTPUT_DIR="./output_infomae"
MODEL="mae_vit_base_patch16"

# Stage 0: Baseline MAE 미세 조정
echo "========================================"
echo "Stage 0: Baseline MAE 미세 조정"
echo "========================================"
python main_pretrain.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}/stage0 \
    --log_dir ${OUTPUT_DIR}/stage0 \
    --model ${MODEL} \
    --batch_size 64 \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --freeze_encoder \
    --unfreeze_last_n_blocks 0

# Stage 1: SWA 추가 (Surprisal-Weighted Attention)
echo "========================================"
echo "Stage 1: SWA 추가"
echo "========================================"
python main_pretrain.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}/stage1 \
    --log_dir ${OUTPUT_DIR}/stage1 \
    --model ${MODEL} \
    --batch_size 64 \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --use_surprisal_attention \
    --surprisal_lambda 1.0 \
    --freeze_encoder \
    --unfreeze_last_n_blocks 0 \
    --resume ${OUTPUT_DIR}/stage0/checkpoint-99.pth

# Stage 2: Adaptive Masking + IB 정규화 추가
echo "========================================"
echo "Stage 2: Adaptive Masking + IB 정규화"
echo "========================================"
python main_pretrain.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}/stage2 \
    --log_dir ${OUTPUT_DIR}/stage2 \
    --model ${MODEL} \
    --batch_size 64 \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --use_surprisal_attention \
    --surprisal_lambda 1.0 \
    --adaptive_masking \
    --use_epoch_cache \
    --cache_precision float32 \
    --adaptive_alpha 0.0 \
    --adaptive_gamma 1.0 \
    --beta_ib 0.02 \
    --freeze_encoder \
    --unfreeze_last_n_blocks 0 \
    --resume ${OUTPUT_DIR}/stage1/checkpoint-99.pth

# Stage 3: 인코더 부분 미세 조정
echo "========================================"
echo "Stage 3: 인코더 부분 미세 조정"
echo "========================================"
python main_pretrain.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}/stage3 \
    --log_dir ${OUTPUT_DIR}/stage3 \
    --model ${MODEL} \
    --batch_size 64 \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --use_surprisal_attention \
    --surprisal_lambda 1.0 \
    --adaptive_masking \
    --use_epoch_cache \
    --cache_precision float32 \
    --adaptive_alpha 0.0 \
    --adaptive_gamma 1.0 \
    --beta_ib 0.02 \
    --unfreeze_last_n_blocks 2 \
    --resume ${OUTPUT_DIR}/stage2/checkpoint-99.pth

echo "========================================"
echo "InfoMAE 4단계 훈련 완료!"
echo "최종 모델: ${OUTPUT_DIR}/stage3/checkpoint-99.pth"
echo "========================================"
