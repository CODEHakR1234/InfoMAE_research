# MAE 파인튜닝 가이드 (ImageNet-100)

이 문서는 MAE 모델을 ImageNet-100으로 파인튜닝하는 방법을 설명합니다.
ImageNet-100은 ImageNet-1K의 100개 클래스 서브셋으로, 빠른 실험과 검증에 적합합니다.

## 1. 환경 설정

### 가상환경 생성 및 패키지 설치 (권장)

자동 환경 설정 스크립트를 사용하여 가상환경을 생성하고 필요한 패키지를 설치합니다:

```bash
bash setup_env.sh
```

이 스크립트는 다음을 수행합니다:
1. `venv` 가상환경 생성
2. 가상환경 활성화
3. 필요한 패키지 설치
4. 패키지 버전 확인

**수동 설정 방법:**
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate  # macOS/Linux
# 또는
# venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

**새 터미널에서 가상환경 활성화:**
```bash
cd /path/to/mae
source venv/bin/activate
```

**주요 패키지:**
- `torch>=1.8.1`
- `torchvision>=0.9.1`
- `timm==0.3.2` (정확한 버전 필수!)
- `numpy`, `Pillow`, `tensorboard`
- `datasets`, `tqdm` (ImageNet-100 다운로드용)

## 2. 체크포인트 다운로드

사전 학습된 MAE 체크포인트를 다운로드합니다:

```bash
bash download_checkpoints.sh
```

또는 수동으로 다운로드:

```bash
mkdir -p checkpoints

# ViT-Base
wget -P checkpoints https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# ViT-Large
wget -P checkpoints https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth

# ViT-Huge
wget -P checkpoints https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth
```

## 3. ImageNet-100 데이터셋 준비

### 방법 1: Hugging Face Datasets 사용 (권장)

가장 간단한 방법으로 Hugging Face에서 ImageNet-100을 다운로드합니다:

```bash
bash download_imagenet100.sh
```

또는 Python 스크립트로 직접 실행:

```bash
python3 download_imagenet100.py --target_dir ./data/imagenet100
```

이 스크립트는:
- Hugging Face Datasets에서 ImageNet-100 다운로드 (약 13GB)
- ImageFolder 형식으로 자동 저장
- `./data/imagenet100/` 디렉토리에 저장

**주의사항:**
- 다운로드에 시간이 오래 걸릴 수 있습니다 (네트워크 속도에 따라)
- 충분한 디스크 공간 필요 (약 13GB)

### 방법 2: ImageNet-1K에서 추출 (ImageNet-1K가 이미 있는 경우)

원본 ImageNet-1K 데이터셋이 있다면 100개 클래스를 추출할 수 있습니다:

```bash
python3 prepare_imagenet100.py \
    --source_dir /path/to/imagenet \
    --target_dir ./data/imagenet100 \
    --symlink  # 또는 --symlink 없이 복사
```

### 데이터셋 구조

다운로드/생성 후 다음 구조가 생성됩니다:

```
./data/imagenet100/
├── train/
│   ├── class_0/
│   │   ├── 000000.JPEG
│   │   ├── 000001.JPEG
│   │   └── ...
│   ├── class_1/
│   └── ... (100개 클래스)
├── val/
│   ├── class_0/
│   ├── class_1/
│   └── ... (100개 클래스)
└── class_info.json  # 메타데이터
```

## 4. 파인튜닝 실행

### 방법 1: 스크립트 사용 (권장)

**단일 GPU 사용 (권장):**
```bash
# 스크립트는 기본적으로 ImageNet-100 경로로 설정되어 있습니다
# 필요시 run_finetune_single_gpu.sh 편집하여 경로 수정
bash run_finetune_single_gpu.sh
```

**다중 GPU 사용:**
```bash
# 필요시 run_finetune.sh 편집하여 경로 수정
bash run_finetune.sh
```

### 방법 2: 직접 명령어 실행

#### 8 GPUs 사용

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --batch_size 32 \
    --accum_iter 4 \
    --model vit_base_patch16 \
    --finetune ./checkpoints/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --nb_classes 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval \
    --data_path ./data/imagenet100 \
    --output_dir ./output_finetune \
    --log_dir ./logs_finetune
```

#### 단일 GPU 사용 (권장)

```bash
python main_finetune.py \
    --batch_size 16 \
    --accum_iter 8 \
    --model vit_base_patch16 \
    --finetune ./checkpoints/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --nb_classes 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --data_path ./data/imagenet100 \
    --output_dir ./output_finetune
```

### 모델별 하이퍼파라미터

#### ViT-Base
- `--model vit_base_patch16`
- `--blr 5e-4 --layer_decay 0.65 --drop_path 0.1`
- `--epochs 100`

#### ViT-Large
- `--model vit_large_patch16`
- `--blr 1e-3 --layer_decay 0.75 --drop_path 0.2`
- `--epochs 50`

#### ViT-Huge
- `--model vit_huge_patch14`
- `--blr 1e-3 --layer_decay 0.75 --drop_path 0.3`
- `--epochs 50`

## 5. 모델 평가

파인튜닝된 모델을 평가합니다:

```bash
python main_finetune.py \
    --eval \
    --resume ./output_finetune/checkpoint-best.pth \
    --model vit_base_patch16 \
    --batch_size 16 \
    --nb_classes 100 \
    --data_path ./data/imagenet100
```

또는 스크립트 사용:
```bash
bash run_eval.sh
```

## 6. 주요 파라미터 설명

### 필수 파라미터
- `--data_path`: ImageNet-100 데이터셋 경로 (기본: ./data/imagenet100)
- `--finetune`: 사전 학습된 체크포인트 경로
- `--model`: 모델 아키텍처 (vit_base_patch16, vit_large_patch16, vit_huge_patch14)
- `--nb_classes`: 클래스 수 (ImageNet-100의 경우 100, 기본값: 100)

### 학습 파라미터
- `--batch_size`: GPU당 배치 크기 (기본: 32)
- `--accum_iter`: 그래디언트 누적 반복 횟수 (메모리 절약용)
- `--epochs`: 학습 에폭 수
- `--blr`: 기본 학습률 (base learning rate)
- `--layer_decay`: 레이어별 학습률 감소율

### 데이터 증강
- `--mixup`: Mixup alpha 값 (0.8 권장)
- `--cutmix`: CutMix alpha 값 (1.0 권장)
- `--reprob`: Random Erase 확률 (0.25 권장)

### 기타
- `--output_dir`: 체크포인트 저장 디렉토리
- `--log_dir`: TensorBoard 로그 디렉토리
- `--dist_eval`: 분산 평가 활성화 (권장)

## 7. 학습 모니터링

TensorBoard로 학습 진행 상황 확인:

```bash
tensorboard --logdir ./logs_finetune
```

## 8. 예상 결과 (ImageNet-100)

ImageNet-100은 ImageNet-1K의 서브셋이므로, 일반적으로 더 높은 정확도를 보입니다:
- **ViT-Base**: 약 85-88% Top-1 정확도 (예상)
- **ViT-Large**: 약 88-91% Top-1 정확도 (예상)
- **ViT-Huge**: 약 90-92% Top-1 정확도 (예상)

*실제 결과는 하이퍼파라미터와 학습 설정에 따라 다를 수 있습니다.*

## 9. 문제 해결

### GPU 메모리 부족
- `--batch_size`를 줄이기 (예: 16 또는 8)
- `--accum_iter`를 늘려서 effective batch size 유지

### 학습률 설정
- 실제 학습률은 `blr * effective_batch_size / 256`로 자동 계산됩니다
- effective_batch_size = batch_size × accum_iter × num_gpus

### 체크포인트 로드 실패
- 체크포인트 파일 경로 확인
- 모델 아키텍처가 체크포인트와 일치하는지 확인

## 10. 전체 워크플로우 요약

ImageNet-100으로 MAE 파인튜닝의 전체 프로세스:

```bash
# 1. 환경 설정
bash setup_env.sh

# 2. 체크포인트 다운로드
bash download_checkpoints.sh

# 3. ImageNet-100 다운로드
bash download_imagenet100.sh

# 4. 파인튜닝 실행
bash run_finetune_single_gpu.sh  # 단일 GPU
# 또는
bash run_finetune.sh  # 다중 GPU

# 5. 평가
bash run_eval.sh
```

## 11. 추가 정보

- ImageNet-100은 빠른 실험과 검증에 적합합니다
- 전체 ImageNet-1K (1000 클래스)로 학습하려면 `--nb_classes 1000`으로 변경
- 데이터 경로를 `--data_path`로 지정하여 사용 가능

