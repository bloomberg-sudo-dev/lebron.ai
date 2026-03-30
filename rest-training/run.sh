#!/bin/bash
# Simple runner for JRE data pipeline

set -e

cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  echo "📥 Installing dependencies..."
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  echo "✅ Setup complete"
else
  # Activate existing venv
  source venv/bin/activate
fi

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
