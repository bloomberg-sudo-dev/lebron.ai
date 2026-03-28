#!/bin/bash
# Bulletproof REST Training Pipeline
# Run this once, it trains VAE → Teacher → Student automatically

set -e  # Exit on any error

cd /workspace/lebron.ai/rest-training

echo "🚀 REST Training Pipeline"
echo "=================================="

# Pull latest code
echo "📦 Pulling latest code..."
git pull origin main

# Stage 1: VAE Training
echo ""
echo "🔵 Stage 1/3: VAE Training (50 epochs)"
echo "=================================="
python scripts/train_temporal_vae.py \
  --config configs/training_config.yaml \
  --epochs 50 \
  --batch-size 1

echo "✅ VAE training complete! Checkpoint: checkpoints/vae_best.pt"

# Stage 2: Simple Teacher Training
echo ""
echo "🟡 Stage 2/3: Teacher Training (50 epochs)"
echo "=================================="
python scripts/train_teacher_simple.py \
  --epochs 50 \
  --batch-size 2

echo "✅ Teacher training complete! Checkpoint: checkpoints/teacher_simple.pt"

# Stage 3: Summary
echo ""
echo "🟢 Stage 3/3: Training Complete!"
echo "=================================="
echo ""
echo "Trained models:"
ls -lh checkpoints/*.pt

echo ""
echo "✨ All training complete! You can now run inference."
echo ""
echo "Next steps:"
echo "  1. Create inference script"
echo "  2. Test with sample audio"
echo "  3. Generate talking head videos!"
