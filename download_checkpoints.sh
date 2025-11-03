#!/bin/bash

# MAE 사전 학습된 체크포인트 다운로드 스크립트

CHECKPOINT_DIR="./checkpoints"
mkdir -p $CHECKPOINT_DIR

echo "MAE 사전 학습 체크포인트 다운로드 중..."

# ViT-Base 사전 학습 체크포인트
echo "다운로드: ViT-Base pretrained checkpoint..."
wget -P $CHECKPOINT_DIR https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# ViT-Large 사전 학습 체크포인트
echo "다운로드: ViT-Large pretrained checkpoint..."
wget -P $CHECKPOINT_DIR https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth

# ViT-Huge 사전 학습 체크포인트
echo "다운로드: ViT-Huge pretrained checkpoint..."
wget -P $CHECKPOINT_DIR https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth

echo "다운로드 완료! 체크포인트는 $CHECKPOINT_DIR 에 저장되었습니다."

