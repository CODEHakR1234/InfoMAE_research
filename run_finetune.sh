#!/bin/bash

# MAE 파인튜닝 실행 스크립트
# 사용법: bash run_finetune.sh

# 설정 (필요에 따라 수정하세요)
DATA_PATH="/path/to/imagenet"  # ImageNet 데이터 경로로 변경하세요
CHECKPOINT_PATH="./checkpoints/mae_pretrain_vit_base.pth"  # 사용할 체크포인트 경로
OUTPUT_DIR="./output_finetune"
LOG_DIR="./logs_finetune"
MODEL="vit_base_patch16"  # 또는 vit_large_patch16, vit_huge_patch14
BATCH_SIZE=32
ACCUM_ITER=4  # GPU 메모리가 부족하면 증가
EPOCHS=100
NUM_GPUS=8  # 사용할 GPU 개수

# 데이터 경로 확인
if [ ! -d "$DATA_PATH" ]; then
    echo "오류: 데이터 경로가 존재하지 않습니다: $DATA_PATH"
    echo "스크립트의 DATA_PATH 변수를 수정해주세요."
    exit 1
fi

# 체크포인트 확인
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "경고: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_PATH"
    echo "먼저 bash download_checkpoints.sh 를 실행하여 체크포인트를 다운로드하세요."
    read -p "그래도 계속하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 출력 디렉토리 생성
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "=== MAE 파인튜닝 시작 ==="
echo "모델: $MODEL"
echo "체크포인트: $CHECKPOINT_PATH"
echo "데이터: $DATA_PATH"
echo "출력: $OUTPUT_DIR"
echo "=========================="

# 단일 노드 분산 학습 실행
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main_finetune.py \
    --batch_size $BATCH_SIZE \
    --accum_iter $ACCUM_ITER \
    --model $MODEL \
    --finetune $CHECKPOINT_PATH \
    --epochs $EPOCHS \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR

echo "파인튜닝 완료!"

