#!/bin/bash
# Runpod auto-setup script
# Run this on Runpod to start training

set -e

echo "🚀 REST Training Setup for Runpod"
echo "=================================="

# 1. Navigate to workspace
cd /workspace

# 2. Clone repo if not exists
if [ ! -d "rest-training" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/bloomberg-sudo-dev/lebron.ai.git
    cd lebron.ai/rest-training
else
    cd rest-training
    git pull
fi

# 3. Setup Python environment
echo "🐍 Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# 4. Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 5. Create directories
echo "📁 Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p datasets
mkdir -p outputs

# 6. Download datasets (optional)
# echo "⬇️  Downloading datasets..."
# python scripts/download_datasets.py

# 7. Start training
echo "🎯 Starting training pipeline..."

echo "Stage 1: Training Temporal VAE"
python scripts/train_temporal_vae.py \
    --config configs/vae_config.yaml \
    --gpus 1 \
    --batch-size 16 \
    --log-dir logs/vae \
    2>&1 | tee logs/vae_training.log

echo "Stage 2: Training Teacher Model (Non-streaming)"
python scripts/train_teacher.py \
    --config configs/training_config.yaml \
    --gpus 1 \
    --batch-size 8 \
    --log-dir logs/teacher \
    --vae-checkpoint checkpoints/vae_best.pt \
    2>&1 | tee logs/teacher_training.log

echo "Stage 3: Training Student Model (Streaming with ASD)"
python scripts/train_student.py \
    --config configs/training_config.yaml \
    --gpus 1 \
    --batch-size 8 \
    --log-dir logs/student \
    --vae-checkpoint checkpoints/vae_best.pt \
    --teacher-checkpoint checkpoints/teacher_best.pt \
    2>&1 | tee logs/student_training.log

echo "✅ Training complete!"
echo "Checkpoints saved to: checkpoints/"
echo "Logs saved to: logs/"
echo ""
echo "To check progress in real-time:"
echo "  tail -f logs/student_training.log"
echo ""
echo "To run inference:"
echo "  python scripts/inference.py --checkpoint checkpoints/student_best.pt"
