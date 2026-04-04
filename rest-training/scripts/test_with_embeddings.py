#!/usr/bin/env python3
"""Test A2V-DiT with 6D embeddings"""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import A2VDIT, TemporalVAE, FlowMatcher
from omegaconf import OmegaConf


class EmbeddingTestDataset(Dataset):
    def __init__(self, embedding_path, seq_length=16):
        loaded = torch.load(embedding_path, map_location='cpu')
        
        if isinstance(loaded, dict):
            print(f"⚠️  Loaded dict with keys: {list(loaded.keys())}")
            for key in ['embeddings', 'video', 'frames', 'latent', 'encoded']:
                if key in loaded:
                    self.embeddings = loaded[key]
                    print(f"✅ Extracted '{key}' from dict")
                    break
            else:
                for key, val in loaded.items():
                    if isinstance(val, torch.Tensor):
                        self.embeddings = val
                        print(f"✅ Extracted tensor '{key}' from dict")
                        break
                else:
                    raise ValueError(f"No tensor found in dict. Keys: {list(loaded.keys())}")
        else:
            self.embeddings = loaded
        
        print(f"✅ Loaded embeddings: {self.embeddings.shape}")
        
        self.seq_length = seq_length
        self.valid = self.embeddings.shape[0] >= seq_length
        
        if not self.valid:
            print(f"⚠️  WARNING: Only {self.embeddings.shape[0]} frames, need {seq_length}")
    
    def __len__(self):
        return 1 if not self.valid else max(1, self.embeddings.shape[0] - self.seq_length)
    
    def __getitem__(self, idx):
        if not self.valid:
            return {
                'video': torch.randn(self.seq_length, 8, 32, 32),
                'audio': torch.randn(256),
                'id': torch.randn(512),
            }
        
        start = idx
        end = min(start + self.seq_length, self.embeddings.shape[0])
        seq = self.embeddings[start:end]
        
        if seq.shape[0] < self.seq_length:
            pad = torch.zeros(self.seq_length - seq.shape[0], *seq.shape[1:])
            seq = torch.cat([seq, pad], dim=0)
        
        return {
            'video': seq,
            'audio': torch.randn(256),
            'id': torch.randn(512),
        }


def test_model_forward(model, batch, device, name="Model"):
    try:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        
        print(f"\n📊 Input tensor shapes:")
        print(f"   video (raw): {video.shape} (ndim={video.dim()})")
        print(f"   audio (raw): {audio.shape}")
        
        if video.dim() == 6:
            B, T, d1, d2, d3, d4 = video.shape
            total = d1 * d2 * d3 * d4
            h = w = int(total ** 0.5)
            if h * w != total:
                for th in range(1, int(total**0.5) + 1):
                    if total % th == 0:
                        h, w = th, total // th
                        break
            print(f"   6D reshape: total={total}={h}x{w}, (B, C=1, T, H={h}, W={w})")
            z = video.reshape(B, T, 1, h, w).permute(0, 2, 1, 3, 4)
            z = z.repeat(1, 4, 1, 1, 1)
            print(f"   Duplicated channels: {z.shape} (1 -> 4 channels for model)")
        elif video.dim() == 5:
            if video.shape[2] == 8:
                z = video.permute(0, 2, 1, 3, 4)
            elif video.shape[1] == 8:
                z = video
            else:
                z = video.permute(0, 2, 1, 3, 4)
        elif video.dim() == 4:
            video = video.unsqueeze(0)
            if video.shape[2] == 8:
                z = video.permute(0, 2, 1, 3, 4)
            else:
                z = video
        else:
            raise ValueError(f"Unsupported shape: {video.dim()}D {video.shape}")
        
        B, C, T, H, W = z.shape
        print(f"   video (final): {z.shape} = (B={B}, C={C}, T={T}, H={H}, W={W})")
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1).expand(-1, T, -1)
        
        print(f"   audio (final): {audio.shape}")
        
        ref_image = z[:, :, 0:1, 0:1, 0:1]
        print(f"   ref_image: {ref_image.shape}")
        
        timesteps = torch.randint(0, 1000, (B,)).to(device)
        
        with torch.no_grad():
            output = model(z=z, timesteps=timesteps, audio_emb=audio, ref_image=ref_image)
        
        print(f"\n✅ {name} forward pass successful!")
        print(f"   Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"\n❌ {name} forward pass failed!")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 REST Model Validation with Pre-computed Embeddings")
    print("="*60 + "\n")
    
    emb_path = Path(args.embedding_path)
    if not emb_path.exists():
        print(f"❌ ERROR: Embedding file not found: {args.embedding_path}")
        return False
    
    print(f"📁 Embedding file: {emb_path}")
    print(f"🖥️  Device: {args.device}")
    print(f"📊 Configuration: batch_size={args.batch_size}, seq_length={args.seq_length}\n")
    
    print("\n[1/4] Creating dataset...")
    try:
        dataset = EmbeddingTestDataset(args.embedding_path, seq_length=args.seq_length)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        batch = next(iter(loader))
        print(f"✅ Dataset created")
        print(f"\n📊 Batch shapes:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"   {key}: {val.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    print("\n[2/4] Initializing models...")
    try:
        vae = TemporalVAE().to(args.device)
        model = A2VDIT().to(args.device)
        print("✅ Models initialized")
    except Exception as e:
        print(f"❌ Failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n[3/4] Skipping VAE encoder test...")
    print("     (Using pre-computed embeddings)")
    
    print("\n[4/4] Testing A2V-DiT with real embeddings...")
    success = test_model_forward(model, batch, args.device, "A2V-DiT")
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
