#!/bin/bash

# MAE Decoder 학습 스크립트
# Pretrained encoder를 사용하여 decoder를 학습합니다.

set -e

# 기본 설정
MODEL="mae_vit_base_patch16"
DATA_PATH="./data/imagenet100"
PRETRAINED_ENCODER="./checkpoints/mae_pretrain_vit_base.pth"
OUTPUT_DIR="./output_pretrain_decoder"
BATCH_SIZE=64
EPOCHS=100
LR=1e-3
BLR=1e-3
WARMUP_EPOCHS=10
MASK_RATIO=0.75

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --pretrained_encoder)
            PRETRAINED_ENCODER="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --blr)
            BLR="$2"
            shift 2
            ;;
        --warmup_epochs)
            WARMUP_EPOCHS="$2"
            shift 2
            ;;
        --mask_ratio)
            MASK_RATIO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--data_path PATH] [--pretrained_encoder PATH] [--output_dir DIR] [--batch_size N] [--epochs N] [--lr LR] [--blr BLR] [--warmup_epochs N] [--mask_ratio RATIO]"
            exit 1
            ;;
    esac
done

# 체크포인트 확인
if [ ! -f "$PRETRAINED_ENCODER" ]; then
    echo "오류: Pretrained encoder 체크포인트를 찾을 수 없습니다: $PRETRAINED_ENCODER"
    echo "먼저 체크포인트를 다운로드하세요: bash download_checkpoints.sh"
    exit 1
fi

# 데이터셋 확인
if [ ! -d "$DATA_PATH/train" ]; then
    echo "오류: 데이터셋을 찾을 수 없습니다: $DATA_PATH/train"
    echo "ImageNet-100 데이터셋을 다운로드하세요: bash download_imagenet100.sh"
    exit 1
fi

echo "============================================================"
echo "MAE Decoder 학습 시작"
echo "============================================================"
echo "모델: $MODEL"
echo "Pretrained Encoder: $PRETRAINED_ENCODER"
echo "데이터셋: $DATA_PATH"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "배치 크기: $BATCH_SIZE"
echo "에폭: $EPOCHS"
echo "Learning Rate: $LR"
echo "Base LR: $BLR"
echo "Warmup Epochs: $WARMUP_EPOCHS"
echo "마스킹 비율: $MASK_RATIO"
echo "============================================================"
echo ""

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 단일 GPU 학습
python main_pretrain.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --pretrained_encoder $PRETRAINED_ENCODER \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --blr $BLR \
    --lr $LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --mask_ratio $MASK_RATIO \
    --norm_pix_loss \
    --device cuda

echo ""
echo "============================================================"
echo "학습 완료!"
echo "============================================================"
echo "체크포인트 저장 위치: $OUTPUT_DIR"
echo ""
echo "복원 테스트를 실행하려면:"
echo "  python test_pretrained.py --ckpt $OUTPUT_DIR/checkpoint-$(($EPOCHS-1)).pth --model $MODEL"
echo ""

