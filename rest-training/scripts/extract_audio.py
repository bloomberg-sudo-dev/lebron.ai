#!/usr/bin/env python3
"""
Extract and pre-process audio from video files for training
Outputs raw audio (16kHz mono) as .wav files
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import librosa
import numpy as np
from tqdm import tqdm

def extract_audio_ffmpeg(video_path, output_path, sr=16000):
    """Extract audio using ffmpeg"""
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:a", "9",
        "-n",  # Don't overwrite
        "-acodec", "libmp3lame",
        "-ab", "192k",
        "-ar", str(sr),
        "-ac", "1",  # mono
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True
        elif "File already exists" in result.stderr:
            return True
        else:
            print(f"  ❌ ffmpeg failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout extracting audio from {video_path}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def process_videos(video_dir, output_dir, sr=16000):
    """Process all video files in directory"""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv'}
    videos = [f for f in video_dir.glob('*') if f.suffix.lower() in video_extensions]
    
    if not videos:
        print(f"❌ No videos found in {video_dir}")
        return False
    
    print(f"📹 Found {len(videos)} videos")
    
    success_count = 0
    for video_path in tqdm(videos, desc="Extracting audio"):
        output_path = output_dir / f"{video_path.stem}.wav"
        
        if output_path.exists():
            print(f"  ⏭️  {video_path.name} → already extracted")
            success_count += 1
            continue
        
        print(f"  🎵 {video_path.name}...")
        if extract_audio_ffmpeg(video_path, output_path, sr=sr):
            success_count += 1
            print(f"  ✅ {video_path.stem}.wav ({output_path.stat().st_size / 1e6:.1f} MB)")
    
    print(f"\n✅ Extracted {success_count}/{len(videos)} audio files")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Extract audio from videos")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory with video files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for .wav files")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Audio Extraction")
    print("=" * 60)
    
    success = process_videos(args.video_dir, args.output_dir, sr=args.sr)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
