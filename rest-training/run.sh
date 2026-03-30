#!/bin/bash
# Simple runner for JRE data pipeline

set -e

cd "$(dirname "$0")"

# Activate venv
source venv/bin/activate

# Parse command
case "$1" in
  download)
    python3 scripts/download_jre_data.py "${@:2}"
    ;;
  preprocess)
    python3 scripts/preprocess_videos.py "${@:2}"
    ;;
  *)
    echo "Usage: ./run.sh [download|preprocess] [options]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh download --url 'https://www.youtube.com/c/joerogan/videos' --num 10"
    echo "  ./run.sh preprocess --input-dir raw_videos/ --output-dir dataset/"
    echo ""
    echo "For help:"
    echo "  ./run.sh download --help"
    echo "  ./run.sh preprocess --help"
    ;;
esac
