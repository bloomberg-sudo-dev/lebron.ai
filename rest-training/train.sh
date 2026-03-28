#!/bin/bash
# Teacher training launcher with debug mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📍 Working directory: $(pwd)"
echo "🐍 Python version: $(python3 --version)"
echo "🔥 PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')" 2>/dev/null || echo "   (PyTorch not available in this environment)"
echo ""

# Ensure checkpoints dir exists
mkdir -p checkpoints

# Clear pycache to avoid stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Check for flags
if [[ "$1" == "--debug" ]]; then
    echo "🐛 DEBUG MODE: Running one iteration with verbose output..."
    echo ""
    python3 scripts/train_teacher_debug.py \
        --epochs 1 \
        --batch-size 2 \
        --lr 1e-3 \
        --data-root datasets/ \
        --checkpoint-dir checkpoints/
elif [[ "$1" == "--working" ]]; then
    echo "🚀 Running WORKING version with DYNAMIC seq_len..."
    echo ""
    python3 scripts/train_teacher_working.py \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-3 \
        --data-root datasets/ \
        --checkpoint-dir checkpoints/
    
    echo ""
    echo "✅ Training complete! Checkpoint saved to checkpoints/teacher_working.pt"
else
    echo "Usage:"
    echo "  bash train.sh --debug    (test one batch, verbose)"
    echo "  bash train.sh --working  (full training with dynamic seq_len)"
    echo ""
    echo "Running default (full training)..."
    echo ""
    python3 scripts/train_teacher_working.py \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-3 \
        --data-root datasets/ \
        --checkpoint-dir checkpoints/
    
    echo ""
    echo "✅ Training complete! Checkpoint saved to checkpoints/teacher_working.pt"
fi
