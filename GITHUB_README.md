# MAE with ImageNet-100 Support

This repository provides a complete setup for fine-tuning MAE (Masked Autoencoders) on ImageNet-100, including automated dataset download and environment setup scripts.

## Features

✅ **Easy Setup**: Automated environment configuration  
✅ **ImageNet-100 Support**: Automated download from Hugging Face  
✅ **Pre-trained Models**: Easy checkpoint download  
✅ **Single/Multi-GPU**: Support for both single and multi-GPU training  
✅ **Complete Guides**: Step-by-step documentation  

## Quick Start

```bash
# 1. Setup environment
bash setup_env.sh

# 2. Download pre-trained checkpoints
bash download_checkpoints.sh

# 3. Download ImageNet-100 dataset (~13GB)
bash download_imagenet100.sh

# 4. Fine-tune on ImageNet-100
bash run_finetune_single_gpu.sh

# 5. Evaluate
bash run_eval.sh
```

## Documentation

- **[FINETUNE_GUIDE.md](FINETUNE_GUIDE.md)**: Complete fine-tuning guide for ImageNet-100
- **[README_SETUP.md](README_SETUP.md)**: Detailed setup instructions
- **[README.md](README.md)**: Original MAE repository README

## Requirements

- Python 3.8+
- PyTorch >= 1.8.1
- CUDA-capable GPU (recommended)
- ~20GB disk space for ImageNet-100

## What's Included

### Scripts
- `setup_env.sh`: Create virtual environment and install dependencies
- `download_checkpoints.sh`: Download MAE pre-trained models
- `download_imagenet100.sh`: Download ImageNet-100 from Hugging Face
- `run_finetune_single_gpu.sh`: Single-GPU fine-tuning
- `run_finetune.sh`: Multi-GPU fine-tuning
- `run_eval.sh`: Model evaluation

### Python Scripts
- `download_imagenet100.py`: ImageNet-100 download handler
- `check_setup.py`: Environment verification
- `main_finetune.py`: Fine-tuning main script
- `main_pretrain.py`: Pre-training main script

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Reference

Original paper: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)  
Based on: [Facebook Research MAE](https://github.com/facebookresearch/mae)

