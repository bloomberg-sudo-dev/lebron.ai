#!/bin/bash
set -e

# REST Training Pipeline on RunPod
# Stage 1: Extract audio → Stage 2: Train VAE

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VIDEO_DIR="/mnt/persistent/raw_videos"
DATASET_DIR="/mnt/persistent/dataset"
AUDIO_DIR="${DATASET_DIR}/audio"
CHECKPOINT_DIR="/workspace/lebron.ai/rest-training/checkpoints"

echo "=========================================="
echo "REST Training Pipeline"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo "Videos: $VIDEO_DIR"
echo "Audio Output: $AUDIO_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# Step 1: Extract Audio
echo "📹 Step 1: Extracting audio from videos..."
python3 "$SCRIPT_DIR/extract_audio.py" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$AUDIO_DIR" \
    --sr 16000

echo ""
echo "✅ Audio extraction complete"
echo ""

# Step 2: Train VAE (Stage 1)
echo "🎯 Step 2: Training Temporal VAE (Stage 1)..."
python3 "$SCRIPT_DIR/train_stage1_vae.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --data-dir "$DATASET_DIR" \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda \
    --save-every 5

echo ""
echo "✅ Stage 1 (VAE) training complete!"
echo "📊 Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "Next steps:"
echo "  - Stage 2: Train teacher model (A2V-DiT)"
echo "  - Stage 3: Distill to student + streaming"
echo ""
