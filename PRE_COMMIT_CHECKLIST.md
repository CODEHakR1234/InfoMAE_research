# GitHub 업로드 전 체크리스트

## ✅ 완료된 작업

- [x] 환경 설정 스크립트 (`setup_env.sh`)
- [x] 체크포인트 다운로드 스크립트 (`download_checkpoints.sh`)
- [x] ImageNet-100 다운로드 스크립트 (`download_imagenet100.py`, `download_imagenet100.sh`)
- [x] 파인튜닝 실행 스크립트 (`run_finetune.sh`, `run_finetune_single_gpu.sh`)
- [x] 평가 스크립트 (`run_eval.sh`)
- [x] 환경 확인 스크립트 (`check_setup.py`)
- [x] 가이드 문서 (`FINETUNE_GUIDE.md`)
- [x] `.gitignore` 설정 (데이터, 체크포인트, 가상환경 제외)
- [x] `requirements.txt` 업데이트
- [x] README 업데이트

## 📝 알려진 사항

### Lint 경고 (정상)
- `download_imagenet100.py`의 import 경고는 패키지 미설치 시 나타나는 것으로 정상입니다.
- 실제 실행 시 `requirements.txt`의 패키지 설치 후 해결됩니다.

### 파일 권한
- 모든 `.sh` 스크립트는 실행 권한(`+x`)이 설정되어 있습니다.

## 🚀 GitHub 업로드 방법

```bash
cd /Users/ihagmyeong/Documents/VClab/InfoMAE_research

# Git 초기화 (처음인 경우)
git init

# 원격 저장소 추가
git remote add origin <your-github-repo-url>

# 모든 파일 추가
git add mae/

# 커밋
git commit -m "Add MAE with ImageNet-100 support

- Add automated environment setup script
- Add ImageNet-100 download script from Hugging Face
- Add fine-tuning scripts for single/multi-GPU
- Add comprehensive fine-tuning guide
- Update for ImageNet-100 support"

# 푸시
git push -u origin main
```

## 📦 포함된 파일

### 핵심 파일
- `models_mae.py`, `models_vit.py`: 모델 구현
- `main_finetune.py`, `main_pretrain.py`: 메인 스크립트
- `engine_finetune.py`, `engine_pretrain.py`: 학습 엔진
- `util/`: 유틸리티 함수들

### 설정 및 스크립트
- `setup_env.sh`: 환경 설정
- `download_checkpoints.sh`: 체크포인트 다운로드
- `download_imagenet100.sh`, `download_imagenet100.py`: ImageNet-100 다운로드
- `run_finetune.sh`, `run_finetune_single_gpu.sh`: 파인튜닝 실행
- `run_eval.sh`: 평가
- `check_setup.py`: 환경 확인

### 문서
- `README.md`: 메인 README (업데이트됨)
- `FINETUNE_GUIDE.md`: 상세한 파인튜닝 가이드
- `README_SETUP.md`: 설정 가이드
- `GITHUB_README.md`: GitHub용 요약 README

### 설정 파일
- `requirements.txt`: Python 패키지 목록
- `.gitignore`: Git 제외 파일 목록
- `LICENSE`: 라이선스

## ⚠️ 제외된 파일 (`.gitignore`)

다음 파일/폴더는 Git에 포함되지 않습니다:
- `venv/`: 가상환경
- `checkpoints/`: 다운로드된 체크포인트
- `data/`, `imagenet100/`: 데이터셋
- `output_dir/`, `output_finetune/`: 출력 파일
- `logs/`, `logs_finetune/`: 로그 파일
- `*.pth`, `*.pt`: 모델 파일
- `__pycache__/`: Python 캐시

## ✅ 최종 확인

업로드 전 다음을 확인하세요:
1. 모든 스크립트가 실행 권한을 가지고 있는지 (`chmod +x *.sh`)
2. `.gitignore`가 올바르게 설정되어 있는지
3. 개인 정보나 민감한 정보가 포함되지 않았는지
4. README가 명확하게 작성되었는지

