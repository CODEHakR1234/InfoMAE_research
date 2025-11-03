#!/usr/bin/env python
"""
ImageNet-100 데이터셋 다운로드 스크립트
Hugging Face Datasets를 통해 ImageNet-100을 다운로드하고 ImageFolder 형식으로 저장합니다.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm

def download_from_huggingface(target_dir, cache_dir=None):
    """
    Hugging Face Datasets를 통해 ImageNet-100 다운로드 및 저장
    
    Args:
        target_dir: 데이터를 저장할 대상 디렉토리
        cache_dir: Hugging Face 캐시 디렉토리 (None이면 자동 설정)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Hugging Face datasets 라이브러리가 설치되지 않았습니다.")
        print("설치 중...")
        os.system("pip install datasets")
        try:
            from datasets import load_dataset
        except ImportError:
            print("설치 실패. 수동으로 설치해주세요: pip install datasets")
            return False
    
    print("Hugging Face에서 ImageNet-100 다운로드 중...")
    print("이 작업은 시간이 오래 걸릴 수 있습니다.")
    print("(약 13GB 정도의 데이터를 다운로드합니다)")
    
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 캐시 디렉토리 설정
    if cache_dir is None:
        # 캐시 디렉토리를 대상 디렉토리와 같은 위치로 설정 (충분한 공간 보장)
        cache_dir = target_dir.parent / "hf_cache"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Hugging Face 환경 변수 설정 (기본 캐시 사용 방지)
    import os
    os.environ['HF_HOME'] = str(cache_dir.parent)
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
    
    print(f"대상 디렉토리: {target_dir}")
    print(f"캐시 디렉토리: {cache_dir}")
    print(f"Hugging Face 환경 변수 설정됨")
    
    # Hugging Face에서 ImageNet-100 로드
    try:
        print("\n데이터셋 로드 중...")
        # cache_dir을 명시적으로 지정하여 충분한 공간이 있는 위치 사용
        dataset = load_dataset("clane9/imagenet-100", cache_dir=str(cache_dir))
        
        train_dataset = dataset.get('train', None)
        val_dataset = dataset.get('validation', None)
        
        if train_dataset is None:
            print("Train 데이터셋을 찾을 수 없습니다.")
            return False
        
        print(f"\n데이터셋 정보:")
        print(f"  Train 샘플 수: {len(train_dataset)}")
        if val_dataset:
            print(f"  Val 샘플 수: {len(val_dataset)}")
        print(f"  Feature: {train_dataset.features}")
        
        # Train 데이터 저장
        train_dir = target_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTrain 데이터 저장 중: {train_dir}")
        
        # 클래스별 폴더 생성 및 이미지 저장
        class_to_idx = {}
        idx_to_class = {}
        
        for idx, sample in enumerate(tqdm(train_dataset, desc="Train 저장")):
            image = sample['image']
            label = sample['label']
            label_name = sample.get('label_name', f'class_{label}')
            
            # 클래스 폴더 생성
            if label not in class_to_idx:
                class_to_idx[label] = label_name
                idx_to_class[label_name] = label
                class_dir = train_dir / label_name
                class_dir.mkdir(parents=True, exist_ok=True)
            else:
                class_dir = train_dir / label_name
            
            # 이미지 저장
            image_filename = f"{idx:06d}.JPEG"
            image_path = class_dir / image_filename
            
            if isinstance(image, Image.Image):
                image.save(image_path, 'JPEG')
            else:
                # numpy array인 경우
                Image.fromarray(image).save(image_path, 'JPEG')
        
        # Val 데이터 저장
        if val_dataset:
            val_dir = target_dir / "val"
            val_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nVal 데이터 저장 중: {val_dir}")
            
            for idx, sample in enumerate(tqdm(val_dataset, desc="Val 저장")):
                image = sample['image']
                label = sample['label']
                label_name = sample.get('label_name', f'class_{label}')
                
                # 클래스 폴더 생성
                class_dir = val_dir / label_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # 이미지 저장
                image_filename = f"{idx:06d}.JPEG"
                image_path = class_dir / image_filename
                
                if isinstance(image, Image.Image):
                    image.save(image_path, 'JPEG')
                else:
                    Image.fromarray(image).save(image_path, 'JPEG')
        
        # 메타데이터 저장
        meta_path = target_dir / "class_info.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'class_to_idx': class_to_idx,
                'num_classes': len(class_to_idx),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset) if val_dataset else 0
            }, f, indent=2)
        
        print(f"\n✓ 완료!")
        print(f"  저장 위치: {target_dir}")
        print(f"  클래스 수: {len(class_to_idx)}")
        print(f"  Train 이미지: {len(train_dataset)}개")
        if val_dataset:
            print(f"  Val 이미지: {len(val_dataset)}개")
        
        # 캐시 정리 옵션 제안
        print(f"\n참고: Hugging Face 캐시는 {cache_dir}에 저장되었습니다.")
        print(f"필요시 캐시를 삭제하여 공간을 확보할 수 있습니다: rm -rf {cache_dir}")
        
        return True
        
    except OSError as e:
        if "No space left on device" in str(e):
            print(f"\n디스크 공간 부족 오류 발생!")
            print(f"캐시 디렉토리: {cache_dir}")
            print(f"\n해결 방법:")
            print(f"1. 다른 위치에 캐시 저장:")
            print(f"   python3 download_imagenet100.py --target_dir ./data/imagenet100 --cache_dir /path/to/large/disk/hf_cache")
            print(f"2. 기존 Hugging Face 캐시 정리:")
            print(f"   rm -rf ~/.cache/huggingface/datasets")
            print(f"3. ImageNet-1K에서 ImageNet-100 추출 (이미 ImageNet-1K가 있는 경우)")
            return False
        else:
            print(f"\n다운로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            print("\n대안:")
            print("1. 네트워크 연결 확인")
            print("2. ImageNet-1K에서 ImageNet-100을 추출: prepare_imagenet100.py 사용")
            print("3. GitHub 저장소의 스크립트 사용")
            return False
    except Exception as e:
        print(f"\n다운로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n대안:")
        print("1. 네트워크 연결 확인")
        print("2. ImageNet-1K에서 ImageNet-100을 추출: prepare_imagenet100.py 사용")
        print("3. GitHub 저장소의 스크립트 사용")
        return False


def download_from_github_script(target_dir):
    """
    GitHub 저장소의 스크립트를 사용하여 ImageNet-100 생성
    """
    print("\nGitHub 저장소 방법:")
    print("1. https://github.com/danielchyeh/ImageNet-100-Pytorch 클론")
    print("2. generate_IN100.py 스크립트 사용")
    print("3. 원본 ImageNet-1K 데이터셋 필요")
    
    print("\n또는 다음 명령어로 클론:")
    print("git clone https://github.com/danielchyeh/ImageNet-100-Pytorch.git")
    print("cd ImageNet-100-Pytorch")
    print("python generate_IN100.py --source /path/to/imagenet --target {}".format(target_dir))


def main():
    parser = argparse.ArgumentParser(description='ImageNet-100 다운로드')
    parser.add_argument('--target_dir', type=str, default='./data/imagenet100',
                        help='ImageNet-100을 저장할 경로 (기본: ./data/imagenet100)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Hugging Face 캐시 디렉토리 (기본: 대상 디렉토리와 같은 위치의 hf_cache)')
    parser.add_argument('--method', type=str, choices=['huggingface', 'github'],
                        default='huggingface', help='다운로드 방법 선택')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ImageNet-100 다운로드")
    print("=" * 60)
    print(f"대상 경로: {args.target_dir}")
    if args.cache_dir:
        print(f"캐시 경로: {args.cache_dir}")
    print(f"방법: {args.method}")
    print()
    
    if args.method == 'huggingface':
        if download_from_huggingface(args.target_dir, args.cache_dir):
            print("\n" + "=" * 60)
            print("다운로드 완료!")
            print("=" * 60)
            print(f"\n다음 단계:")
            print(f"1. 파인튜닝 스크립트에서 DATA_PATH를 '{args.target_dir}'로 설정")
            print(f"2. bash run_finetune_single_gpu.sh 실행")
            return 0
        else:
            print("\n다운로드 실패했습니다.")
            return 1
    
    elif args.method == 'github':
        download_from_github_script(args.target_dir)
        return 0
    
    return 0


if __name__ == '__main__':
    exit(main())

