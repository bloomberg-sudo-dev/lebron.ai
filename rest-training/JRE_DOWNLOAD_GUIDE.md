# Joe Rogan Experience Dataset Download & Preprocessing

Complete workflow to download JRE episodes and prepare them for fine-tuning the teacher model.

---

## Quick Start

```bash
cd /workspace/lebron.ai/rest-training

# Step 1: Download JRE episodes
python3 scripts/download_jre_data.py \
    --playlist-url "https://www.youtube.com/c/joerogan/videos" \
    --num-episodes 10 \
    --output-dir raw_videos/

# Step 2: Preprocess (crop faces, extract audio)
python3 scripts/preprocess_videos.py \
    --input-dir raw_videos/ \
    --output-dir dataset/ \
    --face-size 512

# Step 3: Upload to RunPod
rsync -av dataset/ /mnt/persistent/jre-dataset/
```

---

## Step 1: Download JRE Episodes

### Option A: From Channel (Recommended)

```bash
python3 scripts/download_jre_data.py \
    --playlist-url "https://www.youtube.com/c/joerogan/videos" \
    --num-episodes 20 \
    --output-dir raw_videos/
```

**Note:** Replace `joerogan` with the exact channel if URL changes.

### Option B: From Specific Playlist

```bash
# Get playlist URL from YouTube
# https://www.youtube.com/playlist?list=PLRBN... (example)

python3 scripts/download_jre_data.py \
    --playlist-url "https://www.youtube.com/playlist?list=PLRBN..." \
    --num-episodes 15
```

### Option C: From Episode URLs File

Create `episodes.json`:
```json
[
  "https://www.youtube.com/watch?v=...",
  "https://www.youtube.com/watch?v=...",
  "https://www.youtube.com/watch?v=..."
]
```

Then:
```bash
python3 scripts/download_jre_data.py \
    --episode-urls-file episodes.json \
    --num-episodes 10
```

---

## Step 2: Preprocess Videos

### Extract Face & Audio

```bash
python3 scripts/preprocess_videos.py \
    --input-dir raw_videos/ \
    --output-dir dataset/ \
    --face-size 512 \
    --fps 30
```

**What happens:**
- Extracts frames at 30 FPS
- Crops 512×512 face region (centered)
- Extracts audio to WAV
- Saves to `dataset/{video_name}/`

### Optional: Limit Duration

```bash
# Only process first 30 minutes of each video
python3 scripts/preprocess_videos.py \
    --input-dir raw_videos/ \
    --output-dir dataset/ \
    --duration 1800  # 30 minutes in seconds
```

---

## Step 3: Upload to RunPod

### Option A: Via rsync (If SSH'd into Pod)

```bash
# From local machine
scp -r dataset/ root@{POD_IP}:/mnt/persistent/jre-dataset/
```

### Option B: Via RunPod Persistent Volume

```bash
# Inside RunPod pod
rsync -av /workspace/dataset/ /mnt/persistent/jre-dataset/ --progress
```

### Option C: Compress & Transfer

```bash
# Compress dataset
tar -czf jre_dataset.tar.gz dataset/

# Transfer
scp -P {PORT} jre_dataset.tar.gz root@{POD_IP}:/mnt/persistent/

# Extract on pod
tar -xzf jre_dataset.tar.gz -C /mnt/persistent/
```

---

## Directory Structure

After preprocessing:

```
dataset/
├── episode_1/
│   ├── audio.wav
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── episode_2/
│   ├── audio.wav
│   ├── frame_000000.jpg
│   └── ...
└── ...
```

---

## Storage Requirements

| Item | Size |
|------|------|
| 1 JRE episode (3 hours, original) | ~1-2 GB |
| Processed frames (512×512, JPG) | ~500-800 MB |
| Audio WAV | ~50 MB |
| **Total per episode** | ~600-850 MB |
| **20 episodes** | ~12-17 GB |

---

## Requirements

### Local Machine (for download/preprocess)

```bash
# Install yt-dlp
pip install yt-dlp

# Install ffmpeg
# macOS:
brew install ffmpeg

# Ubuntu/Linux:
sudo apt-get install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

### Python Packages

```bash
pip install opencv-python numpy librosa soundfile torch torchvision
```

---

## Next: Fine-tune Teacher

Once data is on RunPod:

```bash
cd /mnt/persistent/jre-dataset/

python3 /workspace/lebron.ai/rest-training/scripts/train_teacher.py \
    --data-root /mnt/persistent/jre-dataset/ \
    --epochs 50 \
    --batch-size 2 \
    --output-dir /workspace/lebron.ai/rest-training/checkpoints/jre_teacher/
```

**Expected training time:** 24-48 hours on 24GB GPU

---

## Troubleshooting

### "yt-dlp not found"
```bash
pip install yt-dlp --upgrade
```

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg

# Windows: Download from ffmpeg.org
```

### "Permission denied" on RunPod volume
```bash
# Check mount permissions
ls -la /mnt/persistent/

# May need sudo:
sudo chown -R root:root /mnt/persistent/
```

### Download speed too slow
- JRE episodes are large (1-2GB each)
- Consider downloading fewer episodes first (test with 5)
- Run overnight for larger batches

---

## Privacy & Licensing

✅ **JRE Content:** Public YouTube videos  
✅ **Research:** Academic/development use  
✅ **Attribution:** Include source in your project  

⚠️ **Commercial Use:** May require licensing (consult legal)

---

## Tips

1. **Start small:** Test with 3-5 episodes first
2. **Monitor storage:** RunPod volumes cost by GB-month
3. **Compress old:** Delete raw videos after preprocessing
4. **Backup:** Keep best dataset version in persistent volume
5. **Iterate:** Fine-tune with 10 hours, then add more data

---

## Example: Full Workflow

```bash
# Download 15 episodes
python3 scripts/download_jre_data.py \
    --playlist-url "https://www.youtube.com/c/joerogan/videos" \
    --num-episodes 15

# Preprocess
python3 scripts/preprocess_videos.py \
    --input-dir raw_videos/ \
    --output-dir dataset/ \
    --face-size 512

# Check size
du -sh dataset/

# Upload to RunPod persistent volume
scp -r dataset/ root@{POD_IP}:/mnt/persistent/jre-dataset/

# On RunPod: Fine-tune teacher
ssh root@{POD_IP}
python3 train_teacher.py --data-root /mnt/persistent/jre-dataset/
```

**Total time:** ~4-6 hours (download + preprocess + transfer)  
**GPU training:** 24-48 hours

Ready? Start downloading! 🎬
