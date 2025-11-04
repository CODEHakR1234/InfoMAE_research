# InfoMAE utilities for surprisal calculation and adaptive masking
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
import os


class EpochSurprisalCache:
    """
    Epoch-level surprisal cache for adaptive masking.
    Stores surprisal values per image across epochs to enable content-based adaptive masking.
    """

    def __init__(self, cache_dir='./surprisal_cache', precision='float32'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.precision = precision
        self.dtype = torch.float16 if precision == 'float16' else torch.float32

        # Cache format: {image_path: surprisal_tensor}
        self.cache = {}
        self.cache_file = self.cache_dir / 'surprisal_cache.pkl'

        # Load existing cache if available
        self.load_cache()

    def load_cache(self):
        """Load surprisal cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded surprisal cache with {len(self.cache)} images")
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                self.cache = {}
        else:
            print("No existing surprisal cache found, starting fresh")

    def save_cache(self):
        """Save surprisal cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Saved surprisal cache with {len(self.cache)} images")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def update_surprisal(self, image_paths, surprisal_values):
        """
        Update surprisal values for given images

        Args:
            image_paths: List of image paths or indices
            surprisal_values: Tensor of shape (batch_size, num_patches)
        """
        for i, img_path in enumerate(image_paths):
            # Convert tensor path to string if needed
            if isinstance(img_path, torch.Tensor):
                img_path = str(img_path.item())
            elif hasattr(img_path, 'item'):
                img_path = str(img_path.item())
            else:
                img_path = str(img_path)

            self.cache[img_path] = surprisal_values[i].detach().cpu().to(self.dtype)

    def get_surprisal(self, image_paths):
        """
        Get surprisal values for given images

        Args:
            image_paths: List of image paths or indices

        Returns:
            Tensor of shape (batch_size, num_patches) or None if not cached
        """
        surprisals = []
        available_mask = []

        for img_path in image_paths:
            # Convert tensor path to string if needed
            if isinstance(img_path, torch.Tensor):
                img_path = str(img_path.item())
            elif hasattr(img_path, 'item'):
                img_path = str(img_path.item())
            else:
                img_path = str(img_path)

            if img_path in self.cache:
                surprisals.append(self.cache[img_path])
                available_mask.append(True)
            else:
                # Return zero surprisal for uncached images
                surprisals.append(torch.zeros(196, dtype=self.dtype))  # ViT-Base: 196 patches
                available_mask.append(False)

        if surprisals:
            return torch.stack(surprisals), torch.tensor(available_mask)
        return None, None

    def get_cache_stats(self):
        """Get cache statistics"""
        if not self.cache:
            return {"total_images": 0, "memory_usage_mb": 0}

        total_images = len(self.cache)
        # Estimate memory usage (each tensor: 196 * dtype_size)
        dtype_size = 2 if self.precision == 'float16' else 4  # bytes
        memory_usage_mb = total_images * 196 * dtype_size / (1024 * 1024)

        return {
            "total_images": total_images,
            "memory_usage_mb": round(memory_usage_mb, 2),
            "precision": self.precision
        }


def calculate_surprisal(reconstruction_loss, mask, num_patches=196):
    """
    Calculate surprisal values based on reconstruction loss

    Args:
        reconstruction_loss: Per-patch reconstruction loss (batch_size, num_patches)
        mask: Binary mask indicating visible patches (batch_size, num_patches)
        num_patches: Total number of patches

    Returns:
        surprisal: Per-patch surprisal values (batch_size, num_patches)
    """
    # Initialize surprisal with zeros
    surprisal = torch.zeros_like(reconstruction_loss)

    # Calculate surprisal only for masked (invisible) patches
    masked_patches = (mask == 0)  # 0 means masked

    if masked_patches.any():
        # Surprisal is based on reconstruction error for masked patches
        surprisal[masked_patches] = reconstruction_loss[masked_patches]

        # Normalize by patch count to make it comparable across different mask ratios
        patch_count = masked_patches.sum(dim=1, keepdim=True).float()
        surprisal = surprisal / (patch_count + 1e-8)

    return surprisal


def apply_adaptive_masking(base_mask_ratio, surprisal, alpha=0.0, gamma=1.0, num_patches=196):
    """
    Apply adaptive masking based on surprisal values

    Args:
        base_mask_ratio: Base masking ratio (e.g., 0.75)
        surprisal: Per-patch surprisal values (batch_size, num_patches)
        alpha: Offset parameter for sigmoid
        gamma: Scale parameter for sigmoid
        num_patches: Total number of patches

    Returns:
        adaptive_mask: Binary mask (batch_size, num_patches) where 1=keep, 0=mask
    """
    batch_size, _ = surprisal.shape
    device = surprisal.device

    # Calculate adaptive masking probability per patch
    # p_mask = sigmoid(alpha - gamma * S)
    # Higher surprisal -> lower masking probability (more likely to be kept)
    p_mask = torch.sigmoid(alpha - gamma * surprisal)

    # Adjust base masking ratio to maintain overall masking level
    # This ensures the average masking ratio stays close to base_mask_ratio
    avg_p_mask = p_mask.mean(dim=1, keepdim=True)  # (batch_size, 1)
    scaling_factor = base_mask_ratio / (avg_p_mask + 1e-8)
    scaling_factor = torch.clamp(scaling_factor, 0.1, 10.0)  # Prevent extreme scaling

    p_mask = p_mask * scaling_factor

    # Sample mask based on probabilities
    rand_vals = torch.rand(batch_size, num_patches, device=device)
    adaptive_mask = (rand_vals > p_mask).long()  # 1=keep, 0=mask

    return adaptive_mask


def information_bottleneck_loss(latent_z, surprisal_s, beta=0.01):
    """
    Calculate Information Bottleneck regularization loss

    I(Z;S) = H(Z) - H(Z|S)
    where H(Z|S) is conditional entropy, H(Z) is marginal entropy

    We approximate this using variational methods.

    Args:
        latent_z: Latent representations (batch_size, embed_dim)
        surprisal_s: Surprisal values (batch_size, num_patches)
        beta: Weight for IB regularization

    Returns:
        ib_loss: Information bottleneck loss term
    """
    if beta == 0.0:
        return torch.tensor(0.0, device=latent_z.device)

    batch_size = latent_z.shape[0]

    # Approximate mutual information using kernel methods or simple correlation
    # For simplicity, use correlation-based approximation
    # I(Z;S) ≈ mean correlation between Z and S

    # Flatten surprisal to get a scalar per sample
    s_scalar = surprisal_s.mean(dim=1)  # (batch_size,)

    # Calculate correlation between latent features and surprisal
    # This is a simplified approximation of mutual information
    z_mean = latent_z.mean(dim=0, keepdim=True)  # (1, embed_dim)
    s_mean = s_scalar.mean()  # scalar

    z_centered = latent_z - z_mean  # (batch_size, embed_dim)
    s_centered = s_scalar - s_mean  # (batch_size,)

    # Correlation coefficient for each dimension
    z_std = torch.sqrt((z_centered ** 2).mean(dim=0) + 1e-8)  # (embed_dim,)
    s_std = torch.sqrt((s_centered ** 2).mean() + 1e-8)  # scalar

    correlations = (z_centered * s_centered.unsqueeze(1)).mean(dim=0) / (z_std * s_std + 1e-8)

    # Mutual information approximation (absolute correlation sum)
    mi_approx = correlations.abs().sum()

    # IB loss: minimize I(Z;S) to make Z independent of S
    ib_loss = beta * mi_approx

    return ib_loss


def freeze_encoder_blocks(model, unfreeze_last_n=0):
    """
    Freeze encoder blocks, optionally unfreezing the last N blocks

    Args:
        model: MAE model with encoder blocks
        unfreeze_last_n: Number of last blocks to keep trainable
    """
    if hasattr(model, 'blocks'):  # Encoder blocks
        total_blocks = len(model.blocks)
        for i, block in enumerate(model.blocks):
            if i < total_blocks - unfreeze_last_n:
                # Freeze early blocks
                for param in block.parameters():
                    param.requires_grad = False
            else:
                # Keep last N blocks trainable
                for param in block.parameters():
                    param.requires_grad = True

    # Always keep decoder trainable
    if hasattr(model, 'decoder_embed'):
        for param in model.decoder_embed.parameters():
            param.requires_grad = True
    if hasattr(model, 'decoder_blocks'):
        for block in model.decoder_blocks:
            for param in block.parameters():
                param.requires_grad = True
    if hasattr(model, 'decoder_norm'):
        for param in model.decoder_norm.parameters():
            param.requires_grad = True
    if hasattr(model, 'decoder_pred'):
        for param in model.decoder_pred.parameters():
            param.requires_grad = True


def print_infomae_stats(args, cache=None):
    """Print InfoMAE configuration and cache statistics"""
    print("\n=== InfoMAE Configuration ===")
    print(f"Use Surprisal Attention (SWA): {args.use_surprisal_attention}")
    print(f"Adaptive Masking: {args.adaptive_masking}")
    print(f"Use Epoch Cache: {args.use_epoch_cache}")
    print(f"Freeze Encoder: {args.freeze_encoder}")
    if args.freeze_encoder:
        print(f"Unfreeze Last N Blocks: {args.unfreeze_last_n_blocks}")
    print(f"Surprisal Lambda: {args.surprisal_lambda}")
    print(f"Adaptive Alpha: {args.adaptive_alpha}")
    print(f"Adaptive Gamma: {args.adaptive_gamma}")
    print(f"Beta IB: {args.beta_ib}")
    print(f"Cache Precision: {args.cache_precision}")

    if cache is not None:
        stats = cache.get_cache_stats()
        print(f"\nCache Statistics: {stats}")

    # Determine training stage
    if not args.use_surprisal_attention and not args.adaptive_masking and args.freeze_encoder:
        stage = "Stage 0: Baseline MAE 미세 조정"
    elif args.use_surprisal_attention and not args.adaptive_masking and args.freeze_encoder:
        stage = "Stage 1: SWA 추가"
    elif args.use_surprisal_attention and args.adaptive_masking and args.freeze_encoder and args.beta_ib > 0:
        stage = "Stage 2: Adaptive Masking + IB 정규화"
    elif args.use_surprisal_attention and args.adaptive_masking and not args.freeze_encoder and args.beta_ib > 0:
        stage = "Stage 3: 인코더 부분 미세 조정"
    else:
        stage = "Custom Configuration"

    print(f"Training Stage: {stage}")
    print("=" * 50)
