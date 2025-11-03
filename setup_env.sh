#!/bin/bash

# MAE 파인튜닝 환경 설정 스크립트 (가상환경 포함)

VENV_NAME="venv"
VENV_PATH="./${VENV_NAME}"

echo "=== MAE 파인튜닝 환경 설정 ==="

# Python 버전 확인 (최신 트랙: Python 3.11/3.12 권장)
echo "Python 버전 확인 중..."
PYTHON_CMD=""

# Python 3.12 우선 확인
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "✓ Python 3.12를 찾았습니다."
# Python 3.11 확인
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✓ Python 3.11을 찾았습니다."
# Python 3.10 확인
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "✓ Python 3.10을 찾았습니다."
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [ "$PYTHON_VERSION" = "3.12" ] || [ "$PYTHON_VERSION" = "3.11" ] || [ "$PYTHON_VERSION" = "3.10" ]; then
        PYTHON_CMD="python3"
        echo "✓ Python $PYTHON_VERSION을 사용합니다."
    else
        echo "경고: Python 3.11/3.12를 권장합니다. 현재: $(python3 --version)"
        echo "Python 3.12 설치를 시도합니다..."
    fi
fi

# Python 3.10이 없으면 설치 시도
if [ -z "$PYTHON_CMD" ] || [ "$PYTHON_CMD" != "python3.10" ]; then
    echo ""
    echo "Python 3.10 설치 시도 중..."
    
    # OS 감지
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        OS=$DISTRIB_ID
    elif [ -f /etc/debian_version ]; then
        OS=debian
    elif [ -f /etc/SuSe-release ]; then
        OS=suse
    elif [ -f /etc/redhat-release ]; then
        OS=rhel
    else
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    fi
    
    # Ubuntu/Debian 계열
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        echo "Ubuntu/Debian 계열 감지. Python 3.10 설치 중..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y software-properties-common -qq
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update -qq
            sudo apt-get install -y python3.10 python3.10-venv python3.10-dev -qq
            if command -v python3.10 &> /dev/null; then
                PYTHON_CMD="python3.10"
                echo "✓ Python 3.10 설치 완료!"
            fi
        fi
    # CentOS/RHEL 계열
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "fedora" ]; then
        echo "CentOS/RHEL 계열 감지. Python 3.10 설치 중..."
        if command -v yum &> /dev/null; then
            sudo yum install -y python3.10 python3.10-devel -q
            if command -v python3.10 &> /dev/null; then
                PYTHON_CMD="python3.10"
                echo "✓ Python 3.10 설치 완료!"
            fi
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3.10 python3.10-devel -q
            if command -v python3.10 &> /dev/null; then
                PYTHON_CMD="python3.10"
                echo "✓ Python 3.10 설치 완료!"
            fi
        fi
    # macOS
    elif [ "$OS" = "darwin" ] || [ "$(uname)" = "Darwin" ]; then
        echo "macOS 감지. Homebrew를 사용하여 Python 3.10 설치 중..."
        if command -v brew &> /dev/null; then
            brew install python@3.10
            if [ -f /usr/local/bin/python3.10 ] || [ -f /opt/homebrew/bin/python3.10 ]; then
                if [ -f /usr/local/bin/python3.10 ]; then
                    PYTHON_CMD="/usr/local/bin/python3.10"
                else
                    PYTHON_CMD="/opt/homebrew/bin/python3.10"
                fi
                echo "✓ Python 3.10 설치 완료!"
            fi
        else
            echo "경고: Homebrew가 설치되어 있지 않습니다."
            echo "Homebrew 설치: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
    else
        echo "경고: 자동 설치를 지원하지 않는 OS입니다: $OS"
        echo "수동으로 Python 3.10을 설치해주세요."
    fi
fi

# 최종 확인
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "경고: Python 3.10을 찾을 수 없어 python3를 사용합니다."
    else
        echo "오류: Python을 찾을 수 없습니다."
        echo "수동으로 Python 3.10을 설치한 후 다시 시도해주세요."
        exit 1
    fi
fi

echo ""
echo "사용할 Python: $PYTHON_CMD"
$PYTHON_CMD --version

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
        $PYTHON_CMD -m venv "$VENV_PATH"
    else
        echo "기존 가상환경을 사용합니다."
    fi
else
    echo ""
    echo "가상환경 생성 중: $VENV_PATH"
    $PYTHON_CMD -m venv "$VENV_PATH"
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

# timm 버전 체크 (최신 트랙: timm >= 1.0.0)
timm_version=$(python -c "import timm; print(timm.__version__)" 2>/dev/null)
if [ -n "$timm_version" ]; then
    echo "timm 버전: $timm_version (최신 트랙: >= 1.0.0)"
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

