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

# Check for --debug flag
if [[ "$1" == "--debug" ]]; then
    echo "🐛 DEBUG MODE: Running one iteration with verbose output..."
    echo ""
    python3 scripts/train_teacher_debug.py \
        --epochs 1 \
        --batch-size 2 \
        --lr 1e-3 \
        --data-root datasets/ \
        --checkpoint-dir checkpoints/
else
    echo "🚀 Starting teacher training (FIXED version)..."
    echo ""
    python3 scripts/train_teacher_fixed.py \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-3 \
        --data-root datasets/ \
        --checkpoint-dir checkpoints/
    
    echo ""
    echo "✅ Training complete! Checkpoint saved to checkpoints/teacher_fixed.pt"
fi
