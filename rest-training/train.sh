#!/bin/bash
# Robust training launcher with error handling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📍 Working directory: $(pwd)"
echo "🐍 Python version: $(python3 --version)"
echo "🔥 PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# Ensure checkpoints dir exists
mkdir -p checkpoints

# Clear pycache to avoid stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "🚀 Starting teacher training (robust version)..."
echo ""

# Run training
python3 scripts/train_teacher_robust.py \
    --epochs 50 \
    --batch-size 2 \
    --lr 1e-3 \
    --data-root datasets/ \
    --checkpoint-dir checkpoints/

echo ""
echo "✅ Training complete!"
