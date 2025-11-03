#!/bin/bash

# 사전 학습된 모델 평가 스크립트

# 설정
DATA_PATH="/path/to/imagenet"
CHECKPOINT_PATH="./checkpoints/mae_finetuned_vit_base.pth"  # 평가할 체크포인트
MODEL="vit_base_patch16"  # 모델에 맞게 변경: vit_base_patch16, vit_large_patch16, vit_huge_patch14
BATCH_SIZE=16

# 데이터 경로 확인
if [ ! -d "$DATA_PATH" ]; then
    echo "오류: 데이터 경로가 존재하지 않습니다: $DATA_PATH"
    exit 1
fi

# 체크포인트 확인
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "오류: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_PATH"
    exit 1
fi

echo "=== 모델 평가 ==="
echo "모델: $MODEL"
echo "체크포인트: $CHECKPOINT_PATH"
echo "================"

python main_finetune.py \
    --eval \
    --resume $CHECKPOINT_PATH \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --data_path $DATA_PATH

echo "평가 완료!"

