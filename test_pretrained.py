#!/usr/bin/env python
"""
사전 학습된 MAE 모델 복원 결과 확인 스크립트
체크포인트를 로드하고 이미지 복원 결과를 시각화합니다.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from torchvision.datasets import ImageFolder
import random

import models_mae


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    """
    사전 학습된 모델 준비
    """
    # 모델 생성
    model = models_mae.__dict__[arch]()
    
    # 체크포인트 로드
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    print(f"체크포인트 키: {checkpoint.keys()}")
    
    # 모델 가중치 로드
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        msg = model.load_state_dict(checkpoint, strict=False)
    
    print(f"모델 로드 상태: {msg}")
    
    return model


def run_one_image(img, model, mask_ratio=0.75):
    """
    단일 이미지에 대해 MAE 복원 실행
    """
    x = torch.tensor(img)

    # 모델을 평가 모드로
    model.eval()
    
    # 마스킹과 복원 실행
    with torch.no_grad():
        # 이미지 정규화
        x = x.unsqueeze(0)  # 배치 차원 추가
        
        # forward pass
        loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
        
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # 마스크 시각화
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        x = torch.einsum('nchw->nhwc', x)

        # 이미지 정규화 해제
        # MAE는 정규화된 픽셀을 사용하므로 원본과 비교 가능
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        # 복원 결과
        y = y * std + mean
        y = torch.clip(y, 0, 1)
        
        # 원본 이미지
        x = x * std + mean
        x = torch.clip(x, 0, 1)

        # 마스크된 이미지
        im_masked = x * (1 - mask)

        # 복원된 이미지와 마스크된 부분 결합
        im_paste = x * (1 - mask) + y * mask

    return x[0].numpy(), y[0].numpy(), im_masked[0].numpy(), im_paste[0].numpy(), mask[0].numpy()


def main():
    parser = argparse.ArgumentParser(description='사전 학습된 MAE 모델 복원 테스트')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str,
                        help='모델 아키텍처 (mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14)')
    parser.add_argument('--ckpt', default='./checkpoints/mae_pretrain_vit_base.pth', type=str,
                        help='체크포인트 경로')
    parser.add_argument('--image', type=str, default=None,
                        help='테스트할 이미지 경로 (없으면 랜덤 이미지 생성)')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='마스킹 비율 (기본: 0.75)')
    parser.add_argument('--output', type=str, default='./test_output.png',
                        help='출력 이미지 경로')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAE 사전 학습 모델 복원 테스트")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"체크포인트: {args.ckpt}")
    print(f"마스킹 비율: {args.mask_ratio}")
    print()
    
    # 체크포인트 확인
    if not Path(args.ckpt).exists():
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다: {args.ckpt}")
        print("먼저 체크포인트를 다운로드하세요: bash download_checkpoints.sh")
        return 1
    
    # 모델 로드
    print("모델 로드 중...")
    try:
        model = prepare_model(args.ckpt, args.model)
        print("✓ 모델 로드 완료")
    except Exception as e:
        print(f"오류: 모델 로드 실패: {e}")
        return 1
    
    # 이미지 준비
    print("\n이미지 준비 중...")
    if args.image and Path(args.image).exists():
        # 사용자 제공 이미지 사용
        img = Image.open(args.image).convert('RGB')
    else:
        # 랜덤 이미지 생성 (테스트용)
        print("랜덤 이미지 생성 중...")
        img = np.random.rand(224, 224, 3) * 255
        img = Image.fromarray(img.astype(np.uint8))
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    img_array = img_tensor.permute(1, 2, 0).numpy()
    
    print("✓ 이미지 준비 완료")
    
    # 복원 실행
    print("\n복원 실행 중...")
    try:
        x, y, im_masked, im_paste, mask = run_one_image(img_tensor, model, mask_ratio=args.mask_ratio)
        print("✓ 복원 완료")
    except Exception as e:
        print(f"오류: 복원 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 결과 시각화
    print("\n결과 시각화 중...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(x)
    axes[0, 0].set_title('원본 이미지', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(im_masked)
    axes[0, 1].set_title(f'마스크된 이미지 ({args.mask_ratio*100:.0f}% 제거)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(y)
    axes[1, 0].set_title('복원된 이미지', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(im_paste)
    axes[1, 1].set_title('복원 결과 결합', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✓ 결과 저장: {args.output}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"결과 이미지를 확인하세요: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())

