#!/usr/bin/env python
"""
체크포인트 구조 확인 스크립트
"""

import torch
import sys
from pathlib import Path

def check_checkpoint(checkpoint_path):
    """체크포인트의 구조를 확인합니다."""
    if not Path(checkpoint_path).exists():
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    print(f"=" * 60)
    print(f"체크포인트 구조 확인: {checkpoint_path}")
    print(f"=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 최상위 키 확인
    print(f"\n최상위 키: {list(checkpoint.keys())}")
    
    # 모델 가중치 확인
    if 'model' in checkpoint:
        model_dict = checkpoint['model']
        print(f"\n'model' 키의 가중치 수: {len(model_dict.keys())}")
    elif 'state_dict' in checkpoint:
        model_dict = checkpoint['state_dict']
        print(f"\n'state_dict' 키의 가중치 수: {len(model_dict.keys())}")
    else:
        model_dict = checkpoint
        print(f"\n직접 가중치 수: {len(model_dict.keys())}")
    
    # Encoder/Decoder 분류
    encoder_keys = []
    decoder_keys = []
    other_keys = []
    
    for key in model_dict.keys():
        if 'decoder' in key or 'mask_token' in key:
            decoder_keys.append(key)
        elif any(x in key for x in ['patch_embed', 'blocks', 'norm', 'pos_embed', 'cls_token']):
            encoder_keys.append(key)
        else:
            other_keys.append(key)
    
    print(f"\n키 분류:")
    print(f"  Encoder 관련: {len(encoder_keys)}개")
    print(f"  Decoder 관련: {len(decoder_keys)}개")
    print(f"  기타: {len(other_keys)}개")
    
    if encoder_keys:
        print(f"\nEncoder 키 예시 (처음 5개):")
        for key in encoder_keys[:5]:
            print(f"    {key}")
    
    if decoder_keys:
        print(f"\nDecoder 키 예시 (처음 5개):")
        for key in decoder_keys[:5]:
            print(f"    {key}")
    else:
        print(f"\n⚠️  Decoder 키가 없습니다!")
        print(f"    → 이 체크포인트는 encoder만 포함합니다.")
    
    if other_keys:
        print(f"\n기타 키:")
        for key in other_keys:
            print(f"    {key}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python check_checkpoint.py <checkpoint_path>")
        print("예시: python check_checkpoint.py ./checkpoints/mae_pretrain_vit_base.pth")
        sys.exit(1)
    
    check_checkpoint(sys.argv[1])

