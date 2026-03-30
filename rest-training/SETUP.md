# Setup Instructions

## One-Time Setup

After cloning, create the virtual environment:

```bash
cd rest-training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
./run.sh download --url "https://www.youtube.com/c/joerogan/videos" --num 10
./run.sh preprocess --input-dir raw_videos/ --output-dir dataset/
```

## What's in .gitignore

- `venv/` - Virtual environment (recreate locally)
- `raw_videos/` - Downloaded videos
- `dataset/` - Processed frames

