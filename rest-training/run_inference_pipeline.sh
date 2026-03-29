#!/bin/bash
# Complete Inference & Evaluation Pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 REST Model Inference & Evaluation Pipeline"
echo "=============================================="
echo ""

# Ensure directories exist
mkdir -p checkpoints
mkdir -p outputs

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Inference
echo -e "${BLUE}STEP 1: Teacher Model Inference${NC}"
echo "Testing teacher on held-out data and computing quality metrics..."
echo ""
python3 scripts/inference.py \
    --checkpoint-dir checkpoints/ \
    --teacher-ckpt teacher_working.pt \
    --batch-size 4 \
    --num-batches 5 \
    --data-root datasets/

echo ""
echo -e "${GREEN}✅ Inference complete!${NC}\n"

# Step 2: Full Evaluation
echo -e "${BLUE}STEP 2: Comprehensive Evaluation${NC}"
echo "Computing detailed metrics and generating report..."
echo ""
python3 scripts/evaluate.py \
    --checkpoint-dir checkpoints/ \
    --teacher-ckpt teacher_working.pt \
    --batch-size 4 \
    --num-batches 10 \
    --data-root datasets/ \
    --output-report evaluation_report.txt

echo ""
echo -e "${GREEN}✅ Evaluation complete!${NC}\n"

# Step 3: Generate Videos
echo -e "${BLUE}STEP 3: Generate Sample Videos${NC}"
echo "Creating video outputs (real vs predicted)..."
echo ""
python3 scripts/generate_video.py \
    --checkpoint-dir checkpoints/ \
    --teacher-ckpt teacher_working.pt \
    --output-dir outputs/ \
    --batch-size 2 \
    --data-root datasets/ \
    --num-samples 3

echo ""
echo -e "${GREEN}✅ Video generation complete!${NC}\n"

# Summary
echo "=============================================="
echo -e "${GREEN}🎉 PIPELINE COMPLETE!${NC}"
echo "=============================================="
echo ""
echo "📊 Results:"
echo "  - Inference metrics: (computed above)"
echo "  - Evaluation report: evaluation_report.txt"
echo "  - Sample videos: outputs/"
echo ""
echo "Next steps:"
echo "  1. Review evaluation_report.txt for quality metrics"
echo "  2. Check outputs/ for sample videos"
echo "  3. If quality is good, prepare for deployment"
echo ""
