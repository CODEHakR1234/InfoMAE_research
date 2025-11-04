#!/bin/bash

# InfoMAE 단계별 실험 스크립트
# Stage 0: Warmup (encoder freeze, decoder 학습으로 surprisal cache 구축)
# Stage 1-3: 각 기능 실험 (Stage 0의 checkpoint + cache로 시작하여 자체 cache 업데이트)

set -e

# 기본 설정
DATA_PATH="./data/imagenet100"
OUTPUT_DIR="./output_infomae"
MODEL="mae_vit_base_patch16"
SHARED_CACHE_DIR="./shared_surprisal_cache"

# 인자 파싱
STAGE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--stage STAGE_NUM]"
            echo "  STAGE_NUM: 0,1,2,3 or empty for all stages"
            exit 1
            ;;
    esac
done

# Stage 0 실행 조건 확인
if [ -z "$STAGE" ] || [ "$STAGE" = "0" ]; then
echo "========================================"
echo "Stage 0: Baseline (Encoder Freeze - Decoder Warmup)"
echo "========================================"
# 공유 surprisal cache 디렉토리 생성
mkdir -p ${SHARED_CACHE_DIR}

python main_pretrain.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}/stage0 \
    --log_dir ${OUTPUT_DIR}/stage0 \
    --model ${MODEL} \
    --batch_size 64 \
    --epochs 50 \
    --warmup_epochs 10 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --freeze_encoder \
    --use_epoch_cache \
    --cache_precision float32 \
    --resume ./checkpoints/mae_pretrain_vit_base.pth

# Stage 0 완료 후 중간 테스트 실행
echo "========================================"
echo "Stage 0 완료 - 중간 결과 테스트"
echo "========================================"
if [ -f "${OUTPUT_DIR}/stage0/checkpoint-49.pth" ]; then
    ./test_pretrained.sh --ckpt ${OUTPUT_DIR}/stage0/checkpoint-49.pth --output ${OUTPUT_DIR}/stage0/test_stage0.png
    echo "✓ Stage 0 중간 결과 저장: ${OUTPUT_DIR}/stage0/test_stage0.png"
fi

# Stage 0의 surprisal cache를 공유 디렉토리로 복사
if [ -f "${OUTPUT_DIR}/stage0/surprisal_cache/surprisal_cache.pkl" ]; then
    cp ${OUTPUT_DIR}/stage0/surprisal_cache/surprisal_cache.pkl ${SHARED_CACHE_DIR}/
    echo "✓ Stage 0 surprisal cache를 공유 디렉토리로 복사"
fi
fi  # Stage 0 종료

# Stage 1: SWA 실험 (Stage 0 cache 기반)
if [ -z "$STAGE" ] || [ "$STAGE" = "1" ]; then
echo "========================================"
echo "Stage 1: SWA 실험 (Stage 0 cache 기반)"
echo "========================================"
# 공유 surprisal cache를 stage1의 cache 디렉토리로 복사
mkdir -p ${OUTPUT_DIR}/stage1/surprisal_cache
if [ -f "${SHARED_CACHE_DIR}/surprisal_cache.pkl" ]; then
    cp ${SHARED_CACHE_DIR}/surprisal_cache.pkl ${OUTPUT_DIR}/stage1/surprisal_cache/
    echo "✓ 공유 surprisal cache를 Stage 1에 복사"
fi

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
    --resume ${OUTPUT_DIR}/stage0/checkpoint-49.pth

# Stage 1 완료 후 중간 테스트 실행
echo "========================================"
echo "Stage 1 완료 - 중간 결과 테스트"
echo "========================================"
if [ -f "${OUTPUT_DIR}/stage1/checkpoint-99.pth" ]; then
    ./test_pretrained.sh --ckpt ${OUTPUT_DIR}/stage1/checkpoint-99.pth --output ${OUTPUT_DIR}/stage1/test_stage1.png
    echo "✓ Stage 1 중간 결과 저장: ${OUTPUT_DIR}/stage1/test_stage1.png"
fi
fi  # Stage 1 종료

# Stage 2: Adaptive 실험 (Stage 0 cache 기반)
if [ -z "$STAGE" ] || [ "$STAGE" = "2" ]; then
echo "========================================"
echo "Stage 2: Adaptive 실험 (Stage 0 cache 기반)"
echo "========================================"
# 공유 surprisal cache를 stage2의 cache 디렉토리로 복사
mkdir -p ${OUTPUT_DIR}/stage2/surprisal_cache
if [ -f "${SHARED_CACHE_DIR}/surprisal_cache.pkl" ]; then
    cp ${SHARED_CACHE_DIR}/surprisal_cache.pkl ${OUTPUT_DIR}/stage2/surprisal_cache/
    echo "✓ 공유 surprisal cache를 Stage 2에 복사"
fi

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
    --resume ${OUTPUT_DIR}/stage0/checkpoint-49.pth

# Stage 2 완료 후 중간 테스트 실행
echo "========================================"
echo "Stage 2 완료 - 중간 결과 테스트"
echo "========================================"
if [ -f "${OUTPUT_DIR}/stage2/checkpoint-99.pth" ]; then
    ./test_pretrained.sh --ckpt ${OUTPUT_DIR}/stage2/checkpoint-99.pth --output ${OUTPUT_DIR}/stage2/test_stage2.png
    echo "✓ Stage 2 중간 결과 저장: ${OUTPUT_DIR}/stage2/test_stage2.png"
fi
fi  # Stage 2 종료

# Stage 3: Full 실험 (Stage 0 cache 기반)
if [ -z "$STAGE" ] || [ "$STAGE" = "3" ]; then
echo "========================================"
echo "Stage 3: Full 실험 (Stage 0 cache 기반)"
echo "========================================"
# 공유 surprisal cache를 stage3의 cache 디렉토리로 복사
mkdir -p ${OUTPUT_DIR}/stage3/surprisal_cache
if [ -f "${SHARED_CACHE_DIR}/surprisal_cache.pkl" ]; then
    cp ${SHARED_CACHE_DIR}/surprisal_cache.pkl ${OUTPUT_DIR}/stage3/surprisal_cache/
    echo "✓ 공유 surprisal cache를 Stage 3에 복사"
fi

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
    --resume ${OUTPUT_DIR}/stage0/checkpoint-49.pth

# Stage 3 완료 후 최종 테스트 실행
echo "========================================"
echo "Stage 3 완료 - 최종 결과 테스트"
echo "========================================"
if [ -f "${OUTPUT_DIR}/stage3/checkpoint-99.pth" ]; then
    ./test_pretrained.sh --ckpt ${OUTPUT_DIR}/stage3/checkpoint-99.pth --output ${OUTPUT_DIR}/stage3/test_stage3_final.png
    echo "✓ Stage 3 최종 결과 저장: ${OUTPUT_DIR}/stage3/test_stage3_final.png"
fi

echo "========================================"
echo "InfoMAE 단계별 훈련 및 테스트 완료!"
echo ""
echo "📁 Checkpoint 파일들:"
echo "  Stage 0 (Warmup): ${OUTPUT_DIR}/stage0/checkpoint-49.pth"
echo "  Stage 1 (SWA): ${OUTPUT_DIR}/stage1/checkpoint-99.pth"
echo "  Stage 2 (Adaptive): ${OUTPUT_DIR}/stage2/checkpoint-99.pth"
echo "  Stage 3 (Full): ${OUTPUT_DIR}/stage3/checkpoint-99.pth"
echo ""
echo "🖼️  시각화 결과들:"
echo "  Stage 0: ${OUTPUT_DIR}/stage0/test_stage0.png"
echo "  Stage 1: ${OUTPUT_DIR}/stage1/test_stage1.png"
echo "  Stage 2: ${OUTPUT_DIR}/stage2/test_stage2.png"
echo "  Stage 3: ${OUTPUT_DIR}/stage3/test_stage3_final.png"
echo ""
echo "📊 TensorBoard 로그:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/stage0"
echo "  tensorboard --logdir ${OUTPUT_DIR}/stage1"
echo "  tensorboard --logdir ${OUTPUT_DIR}/stage2"
echo "  tensorboard --logdir ${OUTPUT_DIR}/stage3"
echo "========================================"
fi  # Stage 3 종료
