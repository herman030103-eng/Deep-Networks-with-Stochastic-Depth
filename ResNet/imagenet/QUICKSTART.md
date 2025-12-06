# Quick Start Guide: Training ResNet on Full ImageNet

## Prerequisites

1. **Hardware Requirements:**
   - GPU with at least 16GB VRAM (recommended: V100, A100, or RTX 3090/4090)
   - 64GB+ RAM
   - 200GB+ storage for dataset cache (automatic download via Hugging Face)
   - For parallel training: 2 GPUs with 32GB VRAM each

2. **Software Requirements:**
   - Python 3.8+
   - CUDA 11.2+ and cuDNN 8.1+ (for GPU support)
   - TensorFlow 2.12+
   - Hugging Face account (free)

## Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/herman030103-eng/Deep-Networks-with-Stochastic-Depth.git
cd Deep-Networks-with-Stochastic-Depth/ResNet/imagenet

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install tensorflow[and-cuda]>=2.12.0
```

## Step 1: Setup Hugging Face Access

**One-time setup:**

1. Create a free account at https://huggingface.co/
2. Accept dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```

The ImageNet dataset (~150GB) will be automatically downloaded and cached on first use.

## Step 2: Choose Your Training Approach

### Option A: Train Models Separately

**Baseline (No Stochastic Depth):**
```bash
python full_imagenet_train.py \
    --out ./results_baseline \
    --depth 50 \
    --sd 0 \
    --epochs 120 \
    --batch 256
```

**With Stochastic Depth:**
```bash
python full_imagenet_train.py \
    --out ./results_stochastic_depth \
    --depth 50 \
    --sd 1 \
    --pL 0.5 \
    --epochs 120 \
    --batch 256
```

### Option B: Compare Both Models Automatically

**Sequential Training (recommended for single GPU):**
```bash
python compare_training.py \
    --mode sequential \
    --depth 50 \
    --epochs 120 \
    --batch 256
```

**Parallel Training (for dual GPU systems):**
```bash
python compare_training.py \
    --mode parallel \
    --gpu_split \
    --depth 50 \
    --epochs 120 \
    --batch 256
```

## Step 3: Monitor Training Progress

### Using TensorBoard

```bash
# While training is running, open a new terminal
tensorboard --logdir ./results_baseline/tensorboard_logs
# or
tensorboard --logdir ./comparison_results/
```

Then open http://localhost:6006 in your browser.

### Check Training History

```bash
# View training metrics
cat results_baseline/training_history.csv

# View final summary
cat results_baseline/training_summary.csv
```

## Advanced Configuration

### For Best Accuracy

Use ResNet-101 with all modern techniques:

```bash
python full_imagenet_train.py \
    --data_dir /path/to/imagenet \
    --out ./results_best_accuracy \
    --depth 101 \
    --sd 1 \
    --pL 0.5 \
    --mixup 0.2 \
    --label_smoothing 0.1 \
    --epochs 120 \
    --batch 256
```

Expected: ~78-79% Top-1 accuracy

### For Faster Training

Use larger batch size and mixed precision:

```bash
python full_imagenet_train.py \
    --data_dir /path/to/imagenet \
    --out ./results_fast_training \
    --depth 50 \
    --sd 1 \
    --pL 0.5 \
    --batch 512 \
    --epochs 90
```

Mixed precision (FP16) is enabled by default.

### For Memory-Constrained Systems

Reduce batch size:

```bash
python full_imagenet_train.py \
    --data_dir /path/to/imagenet \
    --out ./results_low_memory \
    --depth 50 \
    --sd 1 \
    --pL 0.5 \
    --batch 128 \
    --epochs 120
```

## Expected Training Time

| Model | Hardware | Batch Size | Epochs | Time |
|-------|----------|------------|--------|------|
| ResNet-50 | 1x V100 | 256 | 120 | ~3-4 days |
| ResNet-50 | 1x A100 | 512 | 120 | ~2-3 days |
| ResNet-101 | 1x V100 | 256 | 120 | ~5-6 days |
| ResNet-101 | 1x A100 | 512 | 120 | ~3-4 days |

## Results Analysis

After training completes, compare the results:

```bash
# View summary
cat results_baseline/training_summary.csv
cat results_stochastic_depth/training_summary.csv

# Or if using compare_training.py
cat comparison_results/comparison_*/baseline_no_sd/training_summary.csv
cat comparison_results/comparison_*/stochastic_depth_pL0.5/training_summary.csv
```

Key metrics to compare:
- **val_top1_accuracy**: Top-1 validation accuracy (main metric)
- **val_top5_accuracy**: Top-5 validation accuracy
- **val_loss**: Validation loss
- **training_time_hours**: Total training time

## Troubleshooting

### Out of Memory (OOM) Error

Reduce batch size or use gradient accumulation:
```bash
--batch 128  # or even 64
```

### Slow Training

- Ensure GPU is being used: `nvidia-smi`
- Enable mixed precision (default, already enabled)
- Increase batch size if memory allows
- Use faster GPU (A100 > V100 > RTX 3090)

### Low Accuracy

- Increase training epochs: `--epochs 150`
- Add regularization: `--mixup 0.2 --label_smoothing 0.1`
- Check learning rate (automatically scaled with batch size)

### Interrupted Training

The script automatically saves checkpoints. To resume:
```bash
# Just run the same command again
python full_imagenet_train.py \
    --data_dir /path/to/imagenet \
    --out ./results_baseline \
    ...
```

The script will load `best_model.h5` if it exists.

## Expected Results Summary

### ResNet-50

| Configuration | Top-1 Acc | Improvement |
|---------------|-----------|-------------|
| Baseline | 76.5% | - |
| + Stochastic Depth | 77.2% | +0.7% |
| + SD + MixUp + LS | 77.8% | +1.3% |

### ResNet-101

| Configuration | Top-1 Acc | Improvement |
|---------------|-----------|-------------|
| Baseline | 77.8% | - |
| + Stochastic Depth | 78.5% | +0.7% |
| + SD + MixUp + LS | 79.0% | +1.2% |

## Support

For detailed documentation, see:
- `README_FULL_IMAGENET.md` - Comprehensive documentation
- `full_imagenet_train.py --help` - CLI usage
- `compare_training.py --help` - Comparison tool usage

## Citation

If you use this code, please cite:

```
@inproceedings{huang2016deep,
  title={Deep networks with stochastic depth},
  author={Huang, Gao and Sun, Yu and Liu, Zhuang and Sedra, Daniel and Weinberger, Kilian Q},
  booktitle={European conference on computer vision},
  pages={646--661},
  year={2016}
}
```
