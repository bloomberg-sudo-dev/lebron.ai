#!/usr/bin/env python
"""
Preprocess JRE Videos for Training
- Extract face region
- Normalize audio
- Create training dataset
- Upload to RunPod persistent storage
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import subprocess
import torch
from torchvision import transforms
import librosa
import soundfile as sf


def extract_face_region(frame, face_detector=None, size=512):
    """
    Extract face region from frame
    Uses simple face detection or returns center crop
    """
    h, w = frame.shape[:2]
    
    # Simple center crop (works when face is mostly centered)
    center_y, center_x = h // 2, w // 2
    half_size = size // 2
    
    y1 = max(0, center_y - half_size)
    y2 = min(h, center_y + half_size)
    x1 = max(0, center_x - half_size)
    x2 = min(w, center_x + half_size)
    
    cropped = frame[y1:y2, x1:x2]
    
    # Resize to exact size
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size))
    
    return cropped


def normalize_audio(audio_path, sr=16000):
    """Load and normalize audio"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Normalize to [-1, 1]
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    return y, sr


def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video using ffmpeg"""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "9",
        "-n",  # Don't overwrite
        output_audio_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def process_video(video_path, output_dir, face_size=512, fps=30, duration=None):
    """
    Process single video:
    - Extract frames
    - Extract audio
    - Crop face region
    - Save as training data
    """
    
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎬 Processing: {video_name}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if duration:
        frames_to_process = int(duration * video_fps)
    else:
        frames_to_process = total_frames
    
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {video_fps}")
    print(f"  Processing: {frames_to_process} frames")
    
    # Extract audio
    audio_path = os.path.join(video_output_dir, "audio.wav")
    print(f"  📻 Extracting audio...")
    if extract_audio_from_video(video_path, audio_path):
        print(f"  ✅ Audio extracted")
    else:
        print(f"  ⚠️  Audio extraction failed")
    
    # Process frames
    frame_count = 0
    extracted_frames = 0
    
    while frame_count < frames_to_process:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract face region
        face_frame = extract_face_region(frame, size=face_size)
        
        # Save frame
        frame_path = os.path.join(video_output_dir, f"frame_{extracted_frames:06d}.jpg")
        cv2.imwrite(frame_path, face_frame)
        
        extracted_frames += 1
        frame_count += 1
        
        if extracted_frames % 100 == 0:
            print(f"  Extracted {extracted_frames} frames")
    
    cap.release()
    
    print(f"  ✅ Extracted {extracted_frames} frames")
    print(f"  📁 Saved to: {video_output_dir}")
    
    return extracted_frames, audio_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess Videos for Training")
    parser.add_argument("--input-dir", type=str, default="raw_videos/",
                       help="Directory with downloaded videos")
    parser.add_argument("--output-dir", type=str, default="dataset/",
                       help="Output directory for processed frames")
    parser.add_argument("--face-size", type=int, default=512,
                       help="Face crop size (default 512x512)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second to extract")
    parser.add_argument("--duration", type=float, default=None,
                       help="Max seconds per video (default: full video)")
    parser.add_argument("--upload-to-runpod", action="store_true",
                       help="Upload to RunPod persistent volume")
    parser.add_argument("--runpod-mount", type=str, default="/mnt/persistent/dataset/",
                       help="RunPod mount path")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Find all videos
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    videos = []
    for ext in video_extensions:
        videos.extend(Path(args.input_dir).glob(f"*{ext}"))
    
    print(f"\n🎞️  Found {len(videos)} videos\n")
    
    total_frames = 0
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]")
        frames, audio_path = process_video(
            str(video_path),
            args.output_dir,
            face_size=args.face_size,
            fps=args.fps,
            duration=args.duration
        )
        total_frames += frames
    
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"Total frames extracted: {total_frames}")
    print(f"📁 Dataset saved to: {args.output_dir}")
    
    # Optional: Upload to RunPod
    if args.upload_to_runpod:
        print(f"\n📤 Uploading to RunPod...")
        print(f"   Mount path: {args.runpod_mount}")
        print(f"   Run this on RunPod:")
        print(f"   rsync -av {args.output_dir} {args.runpod_mount}")


if __name__ == "__main__":
    main()
