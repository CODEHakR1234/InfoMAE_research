#!/bin/bash

# 사전 학습된 MAE 모델 복원 테스트 스크립트
# 사용법: ./test_pretrained.sh [--ckpt CHECKPOINT_PATH] [--model MODEL_NAME] [--image IMAGE_PATH]

# 스크립트가 있는 디렉토리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 기본값 설정
CKPT="./checkpoints/mae_pretrain_vit_base.pth"
MODEL="mae_vit_base_patch16"
DATA_PATH="./data/imagenet100"
SPLIT="val"  # train 또는 val
IMAGE=""  # 이미지 경로 (비어있으면 데이터셋에서 선택)
MASK_RATIO=0.75
OUTPUT="./test_reconstruction.png"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --mask_ratio)
            MASK_RATIO="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--ckpt CHECKPOINT] [--model MODEL] [--image IMAGE_PATH] [--data_path PATH] [--split SPLIT] [--mask_ratio RATIO] [--output OUTPUT]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "MAE 사전 학습 모델 복원 테스트"
echo "============================================================"

# 체크포인트 확인
if [ ! -f "$CKPT" ]; then
    echo "체크포인트를 찾을 수 없습니다: $CKPT"
    echo "먼저 체크포인트를 다운로드하세요:"
    echo "  bash download_checkpoints.sh"
    exit 1
fi

# 이미지가 제공된 경우
if [ -n "$IMAGE" ] && [ -f "$IMAGE" ]; then
    python3 test_pretrained.py \
        --model $MODEL \
        --ckpt $CKPT \
        --image "$IMAGE" \
        --mask_ratio $MASK_RATIO \
        --output $OUTPUT
else
    echo "데이터셋에서 이미지를 선택합니다..."
    echo "데이터셋 경로: $DATA_PATH/$SPLIT"
    python3 test_pretrained.py \
        --model $MODEL \
        --ckpt $CKPT \
        --data_path "$DATA_PATH" \
        --split "$SPLIT" \
        --mask_ratio $MASK_RATIO \
        --output $OUTPUT
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "테스트 완료!"
    echo "============================================================"
    echo "결과 이미지: $OUTPUT"
    echo ""
    echo "다음과 같이 직접 실행할 수도 있습니다:"
    echo "  python3 test_pretrained.py --image /path/to/image.jpg"
else
    echo "테스트 실패"
    exit 1
fi

