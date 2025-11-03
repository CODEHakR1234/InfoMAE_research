#!/usr/bin/env python
"""
MAE 파인튜닝 환경 설정 확인 스크립트
필요한 패키지와 설정이 제대로 되어있는지 확인합니다.
"""

import sys
import os

def check_package(package_name, import_name=None, required_version=None):
    """패키지 설치 및 버전 확인"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        
        if required_version and version != required_version:
            print(f"  ⚠ 경고: 요구 버전 {required_version}와 다릅니다!")
            return False
        return True
    except ImportError:
        print(f"✗ {package_name}: 설치되지 않음")
        return False

def check_file(filepath, description):
    """파일 존재 확인"""
    if os.path.exists(filepath):
        print(f"✓ {description}: 존재함")
        return True
    else:
        print(f"✗ {description}: 없음")
        return False

def main():
    print("=" * 50)
    print("MAE 파인튜닝 환경 확인")
    print("=" * 50)
    print()
    
    all_ok = True
    
    # Python 버전 확인
    print("Python 버전:")
    print(f"  {sys.version}")
    print()
    
    # 필수 패키지 확인
    print("필수 패키지 확인:")
    all_ok &= check_package("torch", required_version=None)
    all_ok &= check_package("torchvision", required_version=None)
    all_ok &= check_package("timm", required_version="0.3.2")
    all_ok &= check_package("numpy")
    all_ok &= check_package("PIL", import_name="PIL")
    all_ok &= check_package("tensorboard")
    print()
    
    # CUDA 확인
    print("CUDA 확인:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 버전: {torch.version.cuda}")
        else:
            print("⚠ CUDA 사용 불가 (CPU만 사용 가능)")
    except:
        print("✗ CUDA 확인 실패")
    print()
    
    # 필수 파일 확인
    print("필수 파일 확인:")
    all_ok &= check_file("main_finetune.py", "main_finetune.py")
    all_ok &= check_file("models_vit.py", "models_vit.py")
    all_ok &= check_file("engine_finetune.py", "engine_finetune.py")
    all_ok &= check_file("util/datasets.py", "util/datasets.py")
    print()
    
    # 체크포인트 확인
    print("체크포인트 확인:")
    checkpoints = [
        "checkpoints/mae_pretrain_vit_base.pth",
        "checkpoints/mae_pretrain_vit_large.pth",
        "checkpoints/mae_pretrain_vit_huge.pth",
    ]
    checkpoints_found = False
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            size = os.path.getsize(ckpt) / (1024**3)  # GB
            print(f"✓ {ckpt}: {size:.2f} GB")
            checkpoints_found = True
        else:
            print(f"✗ {ckpt}: 없음")
    
    if not checkpoints_found:
        print("  → 체크포인트를 다운로드하려면: bash download_checkpoints.sh")
    print()
    
    # 요약
    print("=" * 50)
    if all_ok:
        print("✓ 환경 설정이 올바르게 되었습니다!")
    else:
        print("✗ 일부 패키지나 파일이 누락되었습니다.")
        print("  → bash setup_env.sh 를 실행하여 환경을 설정하세요.")
    print("=" * 50)

if __name__ == "__main__":
    main()

