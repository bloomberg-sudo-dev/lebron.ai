#!/bin/bash
# Find embeddings on RunPod server
# Usage: bash scripts/find_embeddings.sh

echo "🔍 Searching for embedding files on RunPod..."
echo ""

# Look for .pt files that might be embeddings
find /workspace -name "*embedding*.pt" 2>/dev/null | head -20

echo ""
echo "Looking for joe rogan specific files..."
find /workspace -name "*joe*" -o -name "*rogan*" 2>/dev/null | head -20

echo ""
echo "All .pt files in lebron.ai/rest-training/:"
find /workspace/lebron.ai/rest-training -name "*.pt" 2>/dev/null | head -20

echo ""
echo "Recent modified .pt files:"
find /workspace -name "*.pt" -type f -mtime -7 2>/dev/null | head -20
