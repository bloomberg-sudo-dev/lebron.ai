#!/usr/bin/env python
"""
Preprocess JRE Videos - Extract Frames & Audio
Simple: Just split videos into frames and extract audio
"""

import os
import cv2
import argparse
import subprocess
from pathlib import Path


def extract_frames(video_path, output_dir, frame_size=512):
    """Extract frames from video and crop center"""
    print(f"\n📹 Processing: {Path(video_path).name}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Frames: {total_frames}, FPS: {fps}")
    
    frame_count = 0
    extracted = 0
    
    while extracted < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Center crop to face_size x face_size
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        size = frame_size // 2
        
        y1 = max(0, cy - size)
        y2 = min(h, cy + size)
        x1 = max(0, cx - size)
        x2 = min(w, cx + size)
        
        cropped = frame[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (frame_size, frame_size))
        
        # Save
        frame_path = os.path.join(output_dir, f"frame_{extracted:06d}.jpg")
        cv2.imwrite(frame_path, cropped)
        
        extracted += 1
        frame_count += 1
        
        if extracted % 100 == 0:
            print(f"    Extracted {extracted} frames")
    
    cap.release()
    print(f"  ✅ Saved {extracted} frames")
    return extracted


def extract_audio(video_path, output_dir):
    """Extract audio from video using ffmpeg (optional, non-blocking)"""
    audio_path = os.path.join(output_dir, "audio.wav")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "9",
        "-n",
        audio_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        print(f"  ✅ Audio extracted")
        return True
    except FileNotFoundError:
        # ffmpeg not installed, skip silently
        return False
    except subprocess.CalledProcessError as e:
        # ffmpeg error, skip silently
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess JRE Videos")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory with MP4 videos")
    parser.add_argument("--output-dir", type=str, default="dataset/",
                       help="Output directory for frames")
    parser.add_argument("--frame-size", type=int, default=512,
                       help="Frame crop size (default 512x512)")
    args = parser.parse_args()
    
    # Find all videos
    videos = list(Path(args.input_dir).glob("*.mp4"))
    
    if not videos:
        print(f"❌ No MP4 files found in {args.input_dir}")
        return
    
    print(f"🎬 Found {len(videos)} videos\n")
    
    total_frames = 0
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video_path.name}")
        
        # Create output directory for this video
        video_name = video_path.stem
        video_out_dir = os.path.join(args.output_dir, video_name)
        Path(video_out_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        frames = extract_frames(str(video_path), video_out_dir, args.frame_size)
        total_frames += frames
        
        # Extract audio
        extract_audio(str(video_path), video_out_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"Total frames: {total_frames}")
    print(f"📁 Output: {args.output_dir}")


if __name__ == "__main__":
    main()
