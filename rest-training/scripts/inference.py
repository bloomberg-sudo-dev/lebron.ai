#!/usr/bin/env python
"""
Teacher Model Inference
Test teacher on held-out data and compute quality metrics
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, SpeechAE
from datasets import TalkingHeadDataLoader


def compute_metrics(pred_latents, target_latents):
    """Compute quality metrics between predicted and target latents"""
    
    # MSE
    mse = nn.MSELoss()(pred_latents, target_latents).item()
    
    # L2 distance
    l2_dist = torch.norm(pred_latents - target_latents, p=2).item()
    
    # Cosine similarity (per sample)
    pred_flat = pred_latents.reshape(pred_latents.shape[0], -1)
    target_flat = target_latents.reshape(target_latents.shape[0], -1)
    
    cos_sim = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1)
    cos_sim_mean = cos_sim.mean().item()
    
    # Correlation
    pred_norm = (pred_flat - pred_flat.mean(dim=0, keepdim=True)) / (pred_flat.std(dim=0, keepdim=True) + 1e-8)
    target_norm = (target_flat - target_flat.mean(dim=0, keepdim=True)) / (target_flat.std(dim=0, keepdim=True) + 1e-8)
    corr = (pred_norm * target_norm).mean().item()
    
    return {
        'mse': mse,
        'l2_dist': l2_dist,
        'cosine_similarity': cos_sim_mean,
        'correlation': corr,
    }


def main():
    parser = argparse.ArgumentParser(description="Teacher Model Inference")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--teacher-ckpt", type=str, default="teacher_working.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--data-root", type=str, default="datasets/")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    # Load VAE (frozen)
    print("📦 Loading VAE...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    try:
        vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae_best.pt", map_location=device))
    except FileNotFoundError:
        print(f"⚠️  VAE checkpoint not found, using random init")
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
    train_loader, val_loader = TalkingHeadDataLoader.create_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print("✅ Data loaded\n")
    
    # Run inference
    print(f"🔍 Running inference on {args.num_batches} batches...\n")
    
    all_metrics = {
        'mse': [],
        'l2_dist': [],
        'cosine_similarity': [],
        'correlation': [],
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.num_batches:
                break
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            # Get VAE latents
            _, vae_latents, _ = vae.encode(video)
            B, C, T, H, W = vae_latents.shape
            target_seq_len = T * H * W
            latents_target = vae_latents.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            
            # Get audio embeddings
            audio_emb = audio_encoder(audio)
            
            # Teacher inference
            pred_latents = teacher(audio_emb, target_seq_len=target_seq_len)
            
            # Compute metrics
            metrics = compute_metrics(pred_latents, latents_target)
            
            for key in metrics:
                all_metrics[key].append(metrics[key])
            
            # Print per-batch
            print(f"Batch {batch_idx+1}/{args.num_batches}:")
            print(f"  MSE:                {metrics['mse']:.6f}")
            print(f"  L2 Distance:        {metrics['l2_dist']:.6f}")
            print(f"  Cosine Similarity:  {metrics['cosine_similarity']:.4f}")
            print(f"  Correlation:        {metrics['correlation']:.4f}")
            print()
    
    # Summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for key in all_metrics:
        values = np.array(all_metrics[key])
        print(f"\n{key.upper()}:")
        print(f"  Mean:   {values.mean():.6f}")
        print(f"  Std:    {values.std():.6f}")
        print(f"  Min:    {values.min():.6f}")
        print(f"  Max:    {values.max():.6f}")
    
    print(f"\n✅ Inference complete!")


if __name__ == "__main__":
    main()
