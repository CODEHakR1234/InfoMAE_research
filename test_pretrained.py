#!/usr/bin/env python
"""
사전 학습된 MAE 모델 복원 결과 확인 스크립트
체크포인트를 로드하고 이미지 복원 결과를 시각화합니다.
"""

# PyTorch 2.6+ compatibility patch for timm
if not hasattr(torch, '_six'):
    import collections.abc as container_abcs
    class _Six:
        container_abcs = container_abcs
    torch._six = _Six()
    import sys
    sys.modules['torch._six'] = torch._six

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
from util.infomae_utils import EpochSurprisalCache


def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', device='cpu'):
    """
    사전 학습된 모델 준비
    """
    # 모델 생성
    model = models_mae.__dict__[arch]()
    
    # 체크포인트 로드 (PyTorch 2.6+ 호환)
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
    print(f"체크포인트 키: {list(checkpoint.keys())}")
    
    # 모델 가중치 로드
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
        print(f"체크포인트 모델 키 수: {len(checkpoint_model.keys())}")
        msg = model.load_state_dict(checkpoint_model, strict=False)
    elif 'state_dict' in checkpoint:
        checkpoint_model = checkpoint['state_dict']
        print(f"체크포인트 state_dict 키 수: {len(checkpoint_model.keys())}")
        msg = model.load_state_dict(checkpoint_model, strict=False)
    else:
        print(f"체크포인트 직접 키 수: {len(checkpoint.keys())}")
        msg = model.load_state_dict(checkpoint, strict=False)
    
    print(f"모델 로드 상태:")
    print(f"  Missing keys: {len(msg.missing_keys)}")
    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    
    # Missing keys가 decoder 관련인지 확인
    decoder_keys = [k for k in msg.missing_keys if 'decoder' in k or 'mask_token' in k]
    encoder_keys = [k for k in msg.missing_keys if 'decoder' not in k and 'mask_token' not in k]
    
    if decoder_keys:
        print(f"  Decoder 관련 missing keys: {len(decoder_keys)}개 (정상 - pretrained checkpoint는 encoder만 포함)")
        print(f"    예: {decoder_keys[:3]}...")
    if encoder_keys:
        print(f"  경고: Encoder 관련 missing keys: {len(encoder_keys)}개")
        print(f"    예: {encoder_keys[:3]}...")
    
    # Decoder가 초기화되지 않았을 수 있으므로 확인
    if len(decoder_keys) > 0:
        print(f"\n  ⚠️  중요: MAE pretrained checkpoint는 encoder만 포함합니다.")
        print(f"  Decoder는 random initialization으로 사용됩니다.")
        print(f"  → 복원 품질이 나쁠 수 있습니다 (decoder가 학습되지 않음).")
        print(f"  → 좋은 복원 결과를 원한다면 pretraining을 완료한 전체 모델을 사용하세요.")
    
    # Load된 encoder 파라미터 수 확인
    loaded_params = sum(p.numel() for name, p in model.named_parameters() 
                       if any(key in name for key in checkpoint_model.keys()))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  파라미터 로드 상태:")
    print(f"    로드된 파라미터: {loaded_params / 1e6:.2f}M / {total_params / 1e6:.2f}M")
    print(f"    로드 비율: {loaded_params / total_params * 100:.1f}%")
    
    model.to(device)
    model.eval()
    
    return model


def run_one_image(img, model, mask_ratio=0.75, device='cpu'):
    """
    단일 이미지에 대해 MAE 복원 실행
    """
    # 이미지가 이미 텐서인 경우
    if isinstance(img, torch.Tensor):
        x = img.clone()
    else:
        x = torch.tensor(img)
    
    # 배치 차원 추가 및 device로 이동
    if x.dim() == 3:
        x = x.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    
    x = x.to(device)
    
    # forward pass
    with torch.no_grad():
        # InfoMAE 모델인지 확인하여 적절한 forward 호출
        if hasattr(model, 'forward') and 'beta_ib' in model.forward.__code__.co_varnames:
            # InfoMAE 모델
            loss, y, mask, surprisal = model(x, mask_ratio=mask_ratio, beta_ib=0.0)
        else:
            # Standard MAE 모델
            loss, y, mask = model(x, mask_ratio=mask_ratio)
            surprisal = None

        loss_value = loss.item()
        print(f"  복원 손실 (loss): {loss_value:.4f}")
        print(f"  참고: 학습된 MAE의 loss는 보통 0.1~0.3 정도입니다.")
        print(f"        현재 loss가 높다면 decoder가 학습되지 않았기 때문입니다.")

        if surprisal is not None:
            avg_surprisal = surprisal.mean().item()
            max_surprisal = surprisal.max().item()
            print(f"  평균 surprisal: {avg_surprisal:.4f}")
            print(f"  최대 surprisal: {max_surprisal:.4f}")
        
        # y는 [N, L, p*p*3] 형태의 patchified 예측
        # unpatchify를 사용하여 이미지로 복원
        y = model.unpatchify(y)  # [N, 3, H, W]
        y = torch.einsum('nchw->nhwc', y).detach().cpu()  # [N, H, W, C]

        # 마스크를 이미지 형태로 변환
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # [N, 3, H, W] - 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()  # [N, H, W, C]
        
        # 원본 이미지도 HWC 형태로 변환
        x = torch.einsum('nchw->nhwc', x.detach().cpu())  # [N, H, W, C]

        # 이미지 정규화 해제 (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        # 복원 결과 정규화 해제
        y = y * std + mean
        y = torch.clip(y, 0, 1)
        
        # 원본 이미지 정규화 해제
        x = x * std + mean
        x = torch.clip(x, 0, 1)

        # 마스크된 이미지 (mask가 1인 부분은 제거됨)
        im_masked = x * (1 - mask)

        # 복원된 이미지와 마스크된 부분 결합
        im_paste = x * (1 - mask) + y * mask

    return x[0].numpy(), y[0].numpy(), im_masked[0].numpy(), im_paste[0].numpy(), mask[0].numpy(), surprisal


def visualize_surprisal(im, mask, surprisal, output_path, title_suffix=""):
    """
    Surprisal 맵 시각화
    """
    if surprisal is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 원본 이미지
    axes[0, 0].imshow(im)
    axes[0, 0].set_title(f'원본 이미지 {title_suffix}')
    axes[0, 0].axis('off')

    # 마스크
    mask_vis = mask[:, :, 0]  # 마스크는 모든 채널이 같음
    axes[0, 1].imshow(mask_vis, cmap='gray')
    axes[0, 1].set_title(f'마스킹 패턴 {title_suffix}')
    axes[0, 1].axis('off')

    # Surprisal 히트맵 (16x16 패치로 리사이즈)
    if isinstance(surprisal, torch.Tensor):
        surprisal_np = surprisal.squeeze().cpu().numpy()
    else:
        surprisal_np = surprisal

    # 196개의 패치를 14x14 그리드로 재배열 (ViT-Base: 224/16 = 14)
    grid_size = int(np.sqrt(len(surprisal_np)))
    surprisal_grid = surprisal_np.reshape(grid_size, grid_size)

    im_surprisal = axes[0, 2].imshow(surprisal_grid, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title(f'Surprisal 맵 {title_suffix}')
    axes[0, 2].axis('off')
    plt.colorbar(im_surprisal, ax=axes[0, 2], shrink=0.8)

    # Surprisal 분포 히스토그램
    axes[1, 0].hist(surprisal_np, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_title(f'Surprisal 분포 {title_suffix}')
    axes[1, 0].set_xlabel('Surprisal 값')
    axes[1, 0].set_ylabel('빈도')

    # Surprisal 통계
    axes[1, 1].text(0.1, 0.8, f'평균: {surprisal_np.mean():.4f}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'표준편차: {surprisal_np.std():.4f}', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'최대값: {surprisal_np.max():.4f}', fontsize=12)
    axes[1, 1].text(0.1, 0.2, f'최소값: {surprisal_np.min():.4f}', fontsize=12)
    axes[1, 1].set_title(f'Surprisal 통계 {title_suffix}')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    # 마스크를 패치 단위로 변환 (224x224 -> 14x14)
    # 각 16x16 패치를 대표하는 값으로 다운샘플링
    patch_size = 16  # ViT-Base 패치 크기
    grid_size = 224 // patch_size  # 14

    # 마스크를 패치별로 변환 (각 패치의 평균값 사용)
    mask_patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch_mask = mask_vis[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            mask_patches.append(patch_mask.mean())
    mask_patches = np.array(mask_patches)  # (196,)

    # 마스킹된 영역 vs 마스킹되지 않은 영역의 surprisal 비교
    masked_surprisal = surprisal_np[mask_patches > 0.5]
    unmasked_surprisal = surprisal_np[mask_patches <= 0.5]

    if len(masked_surprisal) > 0 and len(unmasked_surprisal) > 0:
        axes[1, 2].boxplot([unmasked_surprisal, masked_surprisal],
                          labels=['마스킹되지 않음', '마스킹됨'])
        axes[1, 2].set_title(f'영역별 Surprisal 비교 {title_suffix}')
        axes[1, 2].set_ylabel('Surprisal 값')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Surprisal 분석 결과 저장: {output_path}")


def analyze_infomae_model(model, args):
    """
    InfoMAE 모델 분석
    """
    print("\n=== InfoMAE 모델 분석 ===")

    if hasattr(model, 'use_surprisal_attention') and model.use_surprisal_attention:
        print("✓ Surprisal-Weighted Attention (SWA) 활성화")
        print(f"  Surprisal Lambda: {model.surprisal_lambda}")
    else:
        print("✗ Surprisal-Weighted Attention 비활성화")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능 파라미터 수: {trainable_params:,}")

    # 각 컴포넌트별 파라미터 수
    encoder_params = sum(p.numel() for name, p in model.named_parameters()
                        if 'encoder' in name or 'blocks' in name)
    decoder_params = sum(p.numel() for name, p in model.named_parameters()
                        if 'decoder' in name)

    print(f"  인코더 파라미터: {encoder_params:,}")
    print(f"  디코더 파라미터: {decoder_params:,}")

    print("=" * 30)


def main():
    parser = argparse.ArgumentParser(description='사전 학습된 MAE 모델 복원 테스트')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str,
                        help='모델 아키텍처 (mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14)')
    parser.add_argument('--ckpt', default='./checkpoints/mae_pretrain_vit_base.pth', type=str,
                        help='체크포인트 경로')
    parser.add_argument('--image', type=str, default=None,
                        help='테스트할 이미지 경로 (없으면 데이터셋에서 선택)')
    parser.add_argument('--data_path', type=str, default='./data/imagenet100',
                        help='데이터셋 경로 (ImageFolder 형식)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='데이터셋 split (train 또는 val)')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='마스킹 비율 (기본: 0.75)')
    parser.add_argument('--output', type=str, default='./test_output.png',
                        help='출력 이미지 경로')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드 (재현성을 위해)')
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("MAE 사전 학습 모델 복원 테스트")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"체크포인트: {args.ckpt}")
    print(f"마스킹 비율: {args.mask_ratio}")
    print()
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 체크포인트 확인
    if not Path(args.ckpt).exists():
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다: {args.ckpt}")
        print("먼저 체크포인트를 다운로드하세요: bash download_checkpoints.sh")
        return 1
    
    # 모델 로드
    print("모델 로드 중...")
    try:
        model = prepare_model(args.ckpt, args.model, device=device)
        print("✓ 모델 로드 완료")

        # InfoMAE 모델 분석
        analyze_infomae_model(model, args)
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"오류: 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 이미지 준비
    print("\n이미지 준비 중...")
    class_name = None
    if args.image and Path(args.image).exists():
        # 사용자 제공 이미지 사용
        print(f"사용자 제공 이미지 사용: {args.image}")
        img = Image.open(args.image).convert('RGB')
    else:
        # 데이터셋에서 이미지 선택
        dataset_path = Path(args.data_path) / args.split
        if not dataset_path.exists():
            print(f"오류: 데이터셋 경로를 찾을 수 없습니다: {dataset_path}")
            print("ImageNet-100 데이터셋을 다운로드하세요: bash download_imagenet100.sh")
            print(f"또는 --image 옵션으로 직접 이미지 경로를 지정하세요.")
            return 1
        
        print(f"데이터셋에서 이미지 선택: {dataset_path}")
        
        # ImageFolder 데이터셋 로드 (전처리 없이)
        dataset = ImageFolder(str(dataset_path), transform=None)
        
        if len(dataset) == 0:
            print(f"오류: 데이터셋이 비어있습니다: {dataset_path}")
            return 1
        
        # 랜덤하게 하나 선택
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        class_name = dataset.classes[label]
        
        print(f"선택된 이미지: 인덱스 {idx}/{len(dataset)-1}")
        print(f"클래스: {class_name} (라벨: {label})")
        
        # PIL Image로 변환 (이미 PIL Image일 수도 있음)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
    
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
        x, y, im_masked, im_paste, mask, surprisal = run_one_image(img_tensor, model, mask_ratio=args.mask_ratio, device=device)
        print("✓ 복원 완료")
    except Exception as e:
        print(f"오류: 복원 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 결과 시각화
    print("\n결과 시각화 중...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    title_prefix = f"클래스: {class_name} - " if class_name else ""
    
    axes[0, 0].imshow(x)
    axes[0, 0].set_title(f'{title_prefix}원본 이미지', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(im_masked)
    axes[0, 1].set_title(f'마스크된 이미지 ({args.mask_ratio*100:.0f}% 제거)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(y)
    axes[1, 0].set_title('복원된 이미지', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(im_paste)
    axes[1, 1].set_title('복원 결과 결합', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('MAE 복원 결과', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✓ 결과 저장: {args.output}")

    # InfoMAE: Surprisal 분석 및 시각화
    if surprisal is not None:
        surprisal_output = args.output.replace('.png', '_surprisal.png')
        visualize_surprisal(im_masked, mask, surprisal, surprisal_output, "(InfoMAE)")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    if class_name:
        print(f"테스트 클래스: {class_name}")
    print(f"결과 이미지를 확인하세요: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())

