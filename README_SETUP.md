# MAE Setup and Quick Start Guide

## Overview

This repository is a PyTorch implementation of MAE (Masked Autoencoders) with support for ImageNet-100 fine-tuning. All necessary scripts and configurations are included for easy setup and execution.

## Repository Structure

```
mae/
├── models_mae.py          # MAE model implementation
├── models_vit.py          # Vision Transformer implementation
├── main_finetune.py       # Fine-tuning script
├── main_pretrain.py       # Pre-training script
├── engine_finetune.py     # Fine-tuning engine
├── engine_pretrain.py     # Pre-training engine
├── util/                  # Utility functions
├── setup_env.sh           # Environment setup script
├── download_checkpoints.sh # Download pre-trained checkpoints
├── download_imagenet100.sh # Download ImageNet-100 dataset
├── run_finetune.sh        # Multi-GPU fine-tuning script
├── run_finetune_single_gpu.sh # Single-GPU fine-tuning script
├── run_eval.sh            # Evaluation script
├── check_setup.py         # Environment check script
├── requirements.txt       # Python dependencies
└── FINETUNE_GUIDE.md      # Detailed fine-tuning guide
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment and install dependencies
bash setup_env.sh
```

This will:
- Create a Python virtual environment
- Install all required packages
- Verify installation

### 2. Download Pre-trained Checkpoints

```bash
# Download MAE pre-trained models (ViT-Base, ViT-Large, ViT-Huge)
bash download_checkpoints.sh
```

### 3. Download ImageNet-100 Dataset

```bash
# Download ImageNet-100 from Hugging Face (~13GB)
bash download_imagenet100.sh
```

### 4. Fine-tune on ImageNet-100

```bash
# Single GPU (recommended for testing)
bash run_finetune_single_gpu.sh

# Multi-GPU (8 GPUs)
bash run_finetune.sh
```

### 5. Evaluate

```bash
bash run_eval.sh
```

## Requirements

- Python 3.8+
- PyTorch >= 1.8.1
- CUDA-capable GPU (recommended)
- ~20GB disk space for ImageNet-100

## Supported Models

- ViT-Base (Patch 16)
- ViT-Large (Patch 16)
- ViT-Huge (Patch 14)

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Reference

Original paper: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

This implementation is based on the official [MAE repository](https://github.com/facebookresearch/mae).

