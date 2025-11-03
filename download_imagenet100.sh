#!/bin/bash

# ImageNet-100 다운로드 스크립트 (Hugging Face Datasets)

TARGET_DIR="./data/imagenet100"

echo "============================================================"
echo "ImageNet-100 다운로드 (Hugging Face Datasets)"
echo "============================================================"
echo ""
echo "이 스크립트는 Hugging Face에서 ImageNet-100을 다운로드합니다."
echo "대상 경로: $TARGET_DIR"
echo ""
echo "주의사항:"
echo "  - 약 13GB 정도의 데이터를 다운로드합니다"
echo "  - 시간이 오래 걸릴 수 있습니다 (네트워크 속도에 따라)"
echo "  - 충분한 디스크 공간이 필요합니다"
echo ""

read -p "계속하시겠습니까? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 0
fi

echo ""
echo "다운로드 시작..."
python3 download_imagenet100.py --target_dir "$TARGET_DIR" --method huggingface

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "다운로드 완료!"
    echo "============================================================"
    echo ""
    echo "다음 단계:"
    echo "1. 파인튜닝 스크립트 실행:"
    echo "   bash run_finetune_single_gpu.sh"
    echo ""
    echo "2. 또는 직접 실행:"
    echo "   python main_finetune.py --data_path $TARGET_DIR --nb_classes 100 ..."
else
    echo ""
    echo "다운로드 중 오류가 발생했습니다."
    exit 1
fi

