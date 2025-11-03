#!/bin/bash

# 단일 GPU 파인튜닝 실행 스크립트
# GPU가 1개만 있을 때 사용

# 설정 (필요에 따라 수정하세요)
DATA_PATH="/path/to/imagenet"  # ImageNet 데이터 경로로 변경하세요
CHECKPOINT_PATH="./checkpoints/mae_pretrain_vit_base.pth"
OUTPUT_DIR="./output_finetune"
LOG_DIR="./logs_finetune"
MODEL="vit_base_patch16"
BATCH_SIZE=16
ACCUM_ITER=8  # effective batch = 16 * 8 = 128
EPOCHS=100

# 데이터 경로 확인
if [ ! -d "$DATA_PATH" ]; then
    echo "오류: 데이터 경로가 존재하지 않습니다: $DATA_PATH"
    exit 1
fi

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "=== 단일 GPU 파인튜닝 시작 ==="
echo "모델: $MODEL"
echo "체크포인트: $CHECKPOINT_PATH"
echo "=============================="

# 단일 GPU 학습 (accum_iter로 배치 크기 증가)
python main_finetune.py \
    --batch_size $BATCH_SIZE \
    --accum_iter $ACCUM_ITER \
    --model $MODEL \
    --finetune $CHECKPOINT_PATH \
    --epochs $EPOCHS \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR

echo "파인튜닝 완료!"

