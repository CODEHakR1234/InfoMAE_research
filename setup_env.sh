#!/bin/bash

# MAE 파인튜닝 환경 설정 스크립트 (가상환경 포함)

VENV_NAME="venv"
VENV_PATH="./${VENV_NAME}"

echo "=== MAE 파인튜닝 환경 설정 ==="

# Python 버전 확인
echo "Python 버전 확인 중..."
if ! command -v python3 &> /dev/null; then
    echo "오류: python3를 찾을 수 없습니다."
    exit 1
fi
python3 --version

# 가상환경 생성
if [ -d "$VENV_PATH" ]; then
    echo ""
    echo "가상환경이 이미 존재합니다: $VENV_PATH"
    read -p "기존 가상환경을 사용하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 가상환경 삭제 중..."
        rm -rf "$VENV_PATH"
        echo "새 가상환경 생성 중..."
        python3 -m venv "$VENV_PATH"
    else
        echo "기존 가상환경을 사용합니다."
    fi
else
    echo ""
    echo "가상환경 생성 중: $VENV_PATH"
    python3 -m venv "$VENV_PATH"
fi

# 가상환경 활성화
echo ""
echo "가상환경 활성화 중..."
source "${VENV_PATH}/bin/activate"

# 가상환경 내 Python 경로 확인
echo "가상환경 Python 경로: $(which python)"
echo "가상환경 Python 버전: $(python --version)"

# pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
pip install --upgrade pip --quiet

# 필요한 패키지 설치
echo ""
echo "필요한 패키지 설치 중..."
echo "이 작업은 몇 분 정도 소요될 수 있습니다..."
pip install -r requirements.txt

# timm 버전 확인 (중요!)
echo ""
echo "설치된 패키지 버전 확인:"
python -c "import torch; print(f'torch: {torch.__version__}')" 2>/dev/null || echo "torch: 설치되지 않음"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')" 2>/dev/null || echo "torchvision: 설치되지 않음"
python -c "import timm; print(f'timm: {timm.__version__}')" 2>/dev/null || echo "timm: 설치되지 않음"

# timm 버전 체크
timm_version=$(python -c "import timm; print(timm.__version__)" 2>/dev/null)
if [ "$timm_version" != "0.3.2" ]; then
    echo ""
    echo "경고: timm 버전이 0.3.2가 아닙니다. 현재: $timm_version"
    echo "올바른 버전으로 재설치 중..."
    pip install timm==0.3.2
fi

echo ""
echo "=== 환경 설정 완료 ==="
echo ""
echo "가상환경이 활성화되었습니다."
echo ""
echo "가상환경 사용 방법:"
echo "  활성화: source ${VENV_PATH}/bin/activate"
echo "  비활성화: deactivate"
echo ""
echo "다음 단계:"
echo "1. bash download_checkpoints.sh - 체크포인트 다운로드"
echo "2. run_finetune.sh 스크립트에서 DATA_PATH 설정"
echo "3. 가상환경 활성화 후: bash run_finetune.sh - 파인튜닝 시작"
echo ""
echo "참고: 새로운 터미널을 열 경우 다음 명령어로 가상환경을 활성화하세요:"
echo "  source ${VENV_PATH}/bin/activate"

