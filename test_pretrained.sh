#!/bin/bash

# 사전 학습된 MAE 모델 복원 테스트 스크립트

CKPT="./checkpoints/mae_pretrain_vit_large.pth"
MODEL="mae_vit_large_patch16"
IMAGE=""  # 이미지 경로 (비어있으면 랜덤 이미지 사용)
MASK_RATIO=0.75
OUTPUT="./test_reconstruction.png"

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
    echo "랜덤 이미지로 테스트합니다..."
    python3 test_pretrained.py \
        --model $MODEL \
        --ckpt $CKPT \
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

