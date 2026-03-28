#!/bin/bash
# start-training.sh
# Run this from your laptop to start Runpod training that survives computer sleep
# Usage: ./start-training.sh <runpod-ip-or-host>

if [ -z "$1" ]; then
    echo "Usage: $0 <runpod-ip-or-host>"
    echo ""
    echo "Example:"
    echo "  $0 123.45.67.89"
    echo "  $0 runpod-abc123.com"
    exit 1
fi

POD_HOST="$1"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-1}"

echo "🚀 Starting REST training on Runpod pod: $POD_HOST"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo ""

# SSH into pod and start training with nohup
ssh root@$POD_HOST << 'EOF'

echo "📦 Connected to pod, starting training..."
cd /workspace/lebron.ai/rest-training

# Pull latest code
git pull origin main 2>/dev/null || true

# Start training (survives SSH disconnect + laptop sleep)
nohup python scripts/train_temporal_vae.py \
  --config configs/training_config.yaml \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  > training.log 2>&1 &

PID=$!
echo "✅ Training started with PID: $PID"
echo "   PID: $PID" >> /tmp/training-pid.txt
echo ""
echo "Training is now running in the background."
echo "Your laptop can sleep - Runpod will keep training."
echo ""
echo "To check status later, SSH back in and run:"
echo "  tail -f training.log"
echo "  ps aux | grep python"

EOF

echo ""
echo "✅ Done! SSH connection closed."
echo "Training runs 24/7 on Runpod. Your laptop can now sleep safely."
