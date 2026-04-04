# Training Guide

## Prerequisites

Make sure PyTorch is installed on your RunPod instance:

```bash
# Check if torch is available
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# If not, install (RunPod should have it pre-installed):
pip install torch torchvision
```

## Simple Training Demo

After preprocessing frames:

```bash
python3 scripts/train_simple_demo.py \
  --dataset-dir /mnt/persistent/dataset/ \
  --output-dir /mnt/persistent/models/ \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3
```

**Arguments:**
- `--dataset-dir` (required): Where preprocessed frames are stored
- `--output-dir`: Where to save trained models (default: models/)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32) - reduce if OOM
- `--lr`: Learning rate (default: 1e-3)
- `--max-frames`: Max frames to load for training (default: 10000)

## Training Scripts Available

| Script | Purpose | Status |
|--------|---------|--------|
| `train_simple_demo.py` | ✅ Simple CNN trainer (easiest to start) | **USE THIS** |
| `train_teacher_simple.py` | Advanced (needs model checkpoints) | Skip for now |
| `train_teacher.py` | Full training pipeline | Advanced |
| `train_vae_only.py` | VAE training | Advanced |

## Monitoring Training

In another terminal:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or check outputs
ls -lh /mnt/persistent/models/
```

## Troubleshooting

**"No module named torch"**
```bash
pip install torch torchvision
```

**"CUDA out of memory"**
Reduce batch size:
```bash
--batch-size 16  # or even 8
```

**"No frames found"**
Make sure preprocessing completed:
```bash
ls /mnt/persistent/dataset/ | head -10
```
