#!/usr/bin/env python
"""
Inference Script for REST
Generate video from audio + reference frame
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, A2VDIT, IDContextCache

def main():
    parser = argparse.ArgumentParser(description="REST Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--reference-image", type=str, help="Path to reference image")
    parser.add_argument("--audio-file", type=str, help="Path to audio file")
    parser.add_argument("--output-video", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--num-frames", type=int, default=64, help="Number of frames to generate")
    parser.add_argument("--use-dummy", action="store_true", help="Use dummy reference + audio")
    args = parser.parse_args()
    
    print("=" * 60)
    print("REST: Inference")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Load models
    print(f"\n🧠 Loading models...")
    
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    model = A2VDIT(
        latent_dim=4,
        hidden_dim=768,
        num_heads=12,
        num_blocks=28,
        audio_dim=256,
    ).to(device)
    cache = IDContextCache(latent_dim=4, num_frames=16).to(device)
    
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded from {args.checkpoint}")
    else:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        print("   Using untrained model")
    
    model.eval()
    vae.eval()
    cache.eval()
    
    # Dummy generation
    if args.use_dummy:
        print(f"\n🎬 Generating {args.num_frames} frames (dummy)...")
        
        # Dummy tensors
        with torch.no_grad():
            audio = torch.randn(1, 16000 * 2).to(device)  # 2 seconds
            ref_frame = torch.randn(1, 3, 64, 64).to(device)
            
            # Generate frames
            frames = []
            for i in range(args.num_frames):
                # Forward through model
                latent = model(
                    torch.randn(1, 4, 16, 8, 8).to(device),
                    audio,
                    ref_frame
                )
                frames.append(latent)
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i+1}/{args.num_frames} frames")
            
            print(f"✅ Generation complete!")
            print(f"   Output shape: {latent.shape}")
    
    else:
        print(f"\n🎬 Loading reference and audio...")
        # Would load actual files here
        print("   (Real file loading not implemented in this placeholder)")
    
    print(f"\n✅ Inference complete!")
    print(f"   Output would be saved to: {args.output_video}")

if __name__ == "__main__":
    main()
