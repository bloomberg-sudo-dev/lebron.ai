#!/usr/bin/env python
"""
Download Joe Rogan Experience Episodes for Training
Uses yt-dlp to fetch episodes from YouTube
"""

import os
import subprocess
import argparse
from pathlib import Path
import json


def download_episode(video_url, output_dir):
    """Download single JRE episode"""
    output_path = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",  # Best mp4 format
        "-o", output_path,
        "--quiet",
        "--no-warnings",
        video_url
    ]
    
    try:
        print(f"📥 Downloading: {video_url}")
        subprocess.run(cmd, check=True)
        print(f"✅ Downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download: {e}")
        return False


def get_jre_episodes(playlist_url=None, num_episodes=10):
    """
    Get JRE episode URLs
    
    Default: Latest JRE episodes from official channel
    Playlist: Use specific playlist URL
    """
    
    if playlist_url:
        # User provided playlist
        return [playlist_url]
    
    # Default: Get latest episodes from JRE official channel
    # These are manually curated high-quality episodes
    default_episodes = [
        # Recent, good quality episodes
        "https://www.youtube.com/watch?v=hTMJLUHADiQ",  # Example episode
        # Add more as needed
    ]
    
    print(f"⚠️  Using default playback. For specific episodes, provide --playlist-url")
    print(f"   Example: --playlist-url 'https://www.youtube.com/playlist?list=...'")
    
    return default_episodes[:num_episodes]


def main():
    parser = argparse.ArgumentParser(description="Download JRE Episodes")
    parser.add_argument("--output-dir", type=str, default="raw_videos/")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--playlist-url", type=str, default=None,
                       help="YouTube playlist URL (e.g., channel uploads)")
    parser.add_argument("--episode-urls-file", type=str, default=None,
                       help="JSON file with list of episode URLs")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}\n")
    
    # Get episode URLs
    if args.episode_urls_file and os.path.exists(args.episode_urls_file):
        print(f"📋 Loading episodes from {args.episode_urls_file}")
        with open(args.episode_urls_file, 'r') as f:
            urls = json.load(f)[:args.num_episodes]
    elif args.playlist_url:
        print(f"📋 Using playlist: {args.playlist_url}")
        urls = [args.playlist_url]  # yt-dlp handles playlist expansion
    else:
        print(f"❌ Please provide either:")
        print(f"   --playlist-url 'https://www.youtube.com/c/JRE/videos'")
        print(f"   --episode-urls-file episodes.json")
        print(f"\n📝 To get JRE episodes:")
        print(f"   1. Go to: https://www.youtube.com/c/joerogan/videos")
        print(f"   2. Copy channel URL")
        print(f"   3. Run: python3 download_jre_data.py --playlist-url '<URL>'")
        return
    
    # Download episodes
    print(f"🎬 Downloading {len(urls)} episodes...\n")
    
    successful = 0
    failed = 0
    
    if args.playlist_url:
        # yt-dlp can handle playlists directly
        print(f"📥 Downloading from playlist...")
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]",
            "-o", os.path.join(args.output_dir, "%(title)s.%(ext)s"),
            "--playlist-end", str(args.num_episodes),
            args.playlist_url
        ]
        try:
            subprocess.run(cmd, check=True)
            successful = args.num_episodes
        except subprocess.CalledProcessError as e:
            print(f"❌ Playlist download failed: {e}")
            failed = args.num_episodes
    else:
        # Individual URLs
        for url in urls:
            if download_episode(url, args.output_dir):
                successful += 1
            else:
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Downloaded: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Videos saved to: {args.output_dir}")
    print(f"\nNext step:")
    print(f"  python3 scripts/preprocess_videos.py --input-dir {args.output_dir}")


if __name__ == "__main__":
    main()
