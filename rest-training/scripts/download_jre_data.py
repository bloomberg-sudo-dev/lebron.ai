#!/usr/bin/env python
"""
Download Joe Rogan Experience Episodes for Training
Simple: Just download N episodes from JRE channel
"""

import os
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download JRE Episodes")
    parser.add_argument("--url", type=str, required=True,
                       help="YouTube channel or playlist URL")
    parser.add_argument("--output-dir", type=str, default="raw_videos/",
                       help="Where to save videos")
    parser.add_argument("--num", type=int, default=10,
                       help="Number of episodes to download")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Downloading {args.num} JRE episodes...")
    print(f"📁 Saving to: {args.output_dir}\n")
    
    # Download using yt-dlp
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "-o", os.path.join(args.output_dir, "%(title)s.%(ext)s"),
        "--playlist-end", str(args.num),
        args.url
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Download complete!")
        print(f"📁 Check {args.output_dir} for videos")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"\nMake sure you have yt-dlp installed:")
        print(f"  pip install yt-dlp")
        exit(1)


if __name__ == "__main__":
    main()
