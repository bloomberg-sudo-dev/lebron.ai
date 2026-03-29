#!/usr/bin/env python
"""
Generate Videos from Audio
Use teacher + VAE decoder to create speech-driven video
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, SpeechAE
from datasets import TalkingHeadDataLoader


def latents_to_video(latents, vae, device):
    """
    Decode latents back to video frames
    latents: (B, seq_len, 4) or (B, 4, T, H, W)
    """
    
    # Reshape if needed
    if latents.ndim == 3:
        B, seq_len, C = latents.shape
        # Determine dimensions (assumes square spatial)
        spatial_dim = int(np.sqrt(seq_len))
        T = seq_len // (spatial_dim * spatial_dim)
        latents = latents.reshape(B, T, spatial_dim, spatial_dim, C).permute(0, 4, 1, 2, 3)
    
    # Decode
    with torch.no_grad():
        video_recon = vae.decode(latents)
    
    return video_recon


def save_video(frames, output_path, fps=30):
    """
    Save frames as video file
    frames: (T, H, W, 3) in range [0, 1]
    """
    T, H, W, C = frames.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    for t in range(T):
        frame = (frames[t] * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"✅ Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Videos from Audio")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--teacher-ckpt", type=str, default="teacher_working.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--batch-size", type=int, default=1)  # Reduced default
    parser.add_argument("--data-root", type=str, default="datasets/")
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load VAE
    print("📦 Loading VAE...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    try:
        vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae_best.pt", map_location=device))
    except:
        print("⚠️  VAE checkpoint not found")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("✅ VAE loaded\n")
    
    # Load audio encoder
    print("📦 Loading Audio Encoder...")
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    for param in audio_encoder.parameters():
        param.requires_grad = False
    print("✅ Audio encoder loaded\n")
    
    # Load teacher
    print(f"📦 Loading Teacher ({args.teacher_ckpt})...")
    from train_teacher_working import DynamicTeacher
    teacher = DynamicTeacher(audio_dim=256, latent_dim=4).to(device)
    teacher_path = f"{args.checkpoint_dir}/{args.teacher_ckpt}"
    try:
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    except FileNotFoundError:
        print(f"❌ Teacher checkpoint not found at {teacher_path}")
        sys.exit(1)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"✅ Teacher loaded\n")
    
    # Load data
    print("📊 Loading data...")
    train_loader, _ = TalkingHeadDataLoader.create_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print("✅ Data loaded\n")
    
    # Generate videos
    print(f"🎬 Generating {args.num_samples} videos...\n")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.num_samples:
                break
            
            video_real = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            # Memory cleanup
            torch.cuda.empty_cache()
            
            B = video_real.shape[0]
            
            # Get ground-truth latents
            _, vae_latents_real, _ = vae.encode(video_real)
            B_vae, C, T, H, W = vae_latents_real.shape
            target_seq_len = T * H * W
            
            # Get audio embeddings
            audio_emb = audio_encoder(audio)
            
            # Teacher: audio → predicted latents
            pred_latents = teacher(audio_emb, target_seq_len=target_seq_len)
            
            # Reshape for VAE decoder
            pred_latents_reshaped = pred_latents.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
            
            # Decode: latents → video
            video_pred = latents_to_video(pred_latents, vae, device)
            
            # Save videos (real vs predicted)
            for i in range(B):
                # Real video
                video_real_np = video_real[i].permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
                video_real_np = np.clip(video_real_np, 0, 1)
                save_video(video_real_np, 
                          f"{args.output_dir}/batch{batch_idx}_sample{i}_real.mp4", 
                          fps=30)
                
                # Predicted video
                video_pred_np = video_pred[i].permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
                video_pred_np = np.clip(video_pred_np, 0, 1)
                save_video(video_pred_np, 
                          f"{args.output_dir}/batch{batch_idx}_sample{i}_predicted.mp4", 
                          fps=30)
                
                print(f"✅ Generated batch {batch_idx}, sample {i}")
    
    print(f"\n🎉 All videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
