#!/usr/bin/env python
"""
Full Evaluation Suite
Comprehensive quality assessment of teacher model
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, SpeechAE
from datasets import TalkingHeadDataLoader


class PerceptualMetrics:
    """Compute perceptual quality metrics"""
    
    @staticmethod
    def temporal_coherence(latents, window=5):
        """Measure consistency across time (lower is better)"""
        if latents.shape[1] < window:
            return 0.0
        
        # Compute frame-to-frame differences
        diffs = torch.abs(latents[:, 1:] - latents[:, :-1])
        temporal_var = diffs.mean().item()
        
        return temporal_var
    
    @staticmethod
    def spatial_consistency(latents):
        """Measure spatial smoothness in latent space"""
        # Variance of latents across batch dimension
        spatial_var = latents.std(dim=0).mean().item()
        return spatial_var
    
    @staticmethod
    def reconstruction_error(pred, target):
        """Pixel-space reconstruction error (if videos provided)"""
        if pred.ndim != target.ndim:
            return None
        mse = nn.MSELoss()(pred, target).item()
        return mse


def compute_all_metrics(pred_latents, target_latents, pred_video=None, target_video=None):
    """Compute comprehensive metrics"""
    
    # Latent-space metrics
    metrics = {
        'mse': nn.MSELoss()(pred_latents, target_latents).item(),
        'mae': nn.L1Loss()(pred_latents, target_latents).item(),
        'l2_dist': torch.norm(pred_latents - target_latents, p=2).item(),
    }
    
    # Cosine similarity
    pred_flat = pred_latents.reshape(pred_latents.shape[0], -1)
    target_flat = target_latents.reshape(target_latents.shape[0], -1)
    cos_sim = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1)
    metrics['cosine_similarity'] = cos_sim.mean().item()
    metrics['cosine_similarity_std'] = cos_sim.std().item()
    
    # Correlation
    pred_norm = (pred_flat - pred_flat.mean(dim=0, keepdim=True)) / (pred_flat.std(dim=0, keepdim=True) + 1e-8)
    target_norm = (target_flat - target_flat.mean(dim=0, keepdim=True)) / (target_flat.std(dim=0, keepdim=True) + 1e-8)
    corr = (pred_norm * target_norm).mean().item()
    metrics['correlation'] = corr
    
    # Perceptual metrics
    metrics['temporal_coherence'] = PerceptualMetrics.temporal_coherence(pred_latents)
    metrics['spatial_consistency'] = PerceptualMetrics.spatial_consistency(pred_latents)
    
    # Video reconstruction (if available)
    if pred_video is not None and target_video is not None:
        metrics['video_mse'] = PerceptualMetrics.reconstruction_error(pred_video, target_video)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation Suite")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--teacher-ckpt", type=str, default="teacher_working.pt")
    parser.add_argument("--batch-size", type=int, default=1)  # Reduced default
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="datasets/")
    parser.add_argument("--output-report", type=str, default="evaluation_report.txt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    # Load models
    print("📦 Loading models...")
    
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    try:
        vae.load_state_dict(torch.load(f"{args.checkpoint_dir}/vae_best.pt", map_location=device))
    except:
        print("⚠️  VAE checkpoint not found")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    for param in audio_encoder.parameters():
        param.requires_grad = False
    
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
    
    print("✅ All models loaded\n")
    
    # Load data
    print("📊 Loading data...")
    train_loader, _ = TalkingHeadDataLoader.create_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_dummy=True,
    )
    print("✅ Data loaded\n")
    
    # Evaluate
    print(f"📊 Evaluating on {args.num_batches} batches...\n")
    
    all_metrics = {}
    batch_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.num_batches:
                break
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            # Memory cleanup
            if batch_idx % 2 == 0:
                torch.cuda.empty_cache()
            
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
            batch_metrics = compute_all_metrics(pred_latents, latents_target)
            batch_results.append(batch_metrics)
            
            print(f"Batch {batch_idx+1}: MSE={batch_metrics['mse']:.6f}, CosSim={batch_metrics['cosine_similarity']:.4f}")
    
    # Aggregate statistics
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {args.teacher_ckpt}")
    print(f"Batches evaluated: {len(batch_results)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total samples: {len(batch_results) * args.batch_size}\n")
    
    # Compute aggregate statistics
    report_lines = ["DETAILED RESULTS\n"]
    
    for metric_name in batch_results[0].keys():
        values = np.array([b[metric_name] for b in batch_results])
        
        report_lines.append(f"{metric_name.upper()}:")
        report_lines.append(f"  Mean:       {values.mean():.6f}")
        report_lines.append(f"  Std:        {values.std():.6f}")
        report_lines.append(f"  Min:        {values.min():.6f}")
        report_lines.append(f"  Max:        {values.max():.6f}")
        report_lines.append(f"  Median:     {np.median(values):.6f}\n")
        
        # Print to console
        print(f"{metric_name.upper()}:")
        print(f"  Mean:   {values.mean():.6f}")
        print(f"  Std:    {values.std():.6f}")
        print(f"  Range:  [{values.min():.6f}, {values.max():.6f}]\n")
    
    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    
    mse_values = np.array([b['mse'] for b in batch_results])
    mse_mean = mse_values.mean()
    
    if mse_mean < 0.001:
        quality = "EXCELLENT"
    elif mse_mean < 0.01:
        quality = "GOOD"
    elif mse_mean < 0.05:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    print(f"Overall Quality: {quality}")
    print(f"Mean MSE: {mse_mean:.6f}")
    
    cos_sim_values = np.array([b['cosine_similarity'] for b in batch_results])
    print(f"Mean Cosine Similarity: {cos_sim_values.mean():.4f}")
    print(f"  (Higher is better, range [-1, 1], ideal: 1.0)")
    
    # Save report
    with open(args.output_report, 'w') as f:
        f.write("EVALUATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {args.teacher_ckpt}\n")
        f.write(f"Batches: {len(batch_results)}\n")
        f.write(f"Total samples: {len(batch_results) * args.batch_size}\n\n")
        f.write("\n".join(report_lines))
        f.write(f"\nQuality Assessment: {quality}\n")
    
    print(f"\n✅ Report saved to {args.output_report}")


if __name__ == "__main__":
    main()
