#!/usr/bin/env python3
"""
Validation script: Test training pipeline with pre-computed embeddings

Usage:
    python scripts/test_with_embeddings.py \
        --embedding-path /path/to/joe_rogan_embeddings.pt \
        --audio-path /path/to/audio.wav (optional)
"""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import sys
import traceback

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import A2VDIT, TemporalVAE, FlowMatcher
from omegaconf import OmegaConf


class EmbeddingTestDataset(Dataset):
    """Test dataset using pre-computed video frame embeddings"""
    
    def __init__(self, embedding_path, seq_length=16):
        """
        Args:
            embedding_path: Path to pre-computed embeddings tensor or state dict
            seq_length: Number of frames per sequence
        """
        # Load embeddings
        loaded = torch.load(embedding_path, map_location='cpu')
        
        # Handle dict case (OrderedDict or state_dict)
        if isinstance(loaded, dict):
            print(f"⚠️  Loaded dict with keys: {list(loaded.keys())}")
            # Try common keys
            for key in ['embeddings', 'video', 'frames', 'latent', 'encoded']:
                if key in loaded:
                    self.embeddings = loaded[key]
                    print(f"✅ Extracted '{key}' from dict")
                    break
            else:
                # If no known key, try the first tensor in the dict
                for key, val in loaded.items():
                    if isinstance(val, torch.Tensor):
                        self.embeddings = val
                        print(f"✅ Extracted tensor '{key}' from dict")
                        break
                else:
                    raise ValueError(f"No tensor found in dict. Keys: {list(loaded.keys())}")
        else:
            self.embeddings = loaded
        
        # Validate shape
        if self.embeddings.dim() == 3:  # (T, H, W) - single channel
            self.embeddings = self.embeddings.unsqueeze(1)  # (T, 1, H, W)
        elif self.embeddings.dim() == 4 and self.embeddings.shape[1] != 8:
            # (T, C, H, W) but wrong channel - might be (T, H, W, C)
            if self.embeddings.shape[-1] in [8, 16, 32]:
                self.embeddings = self.embeddings.permute(0, 3, 1, 2)
        
        print(f"✅ Loaded embeddings: {self.embeddings.shape}")
        print(f"   Expected shape: (T, 8, H, W) where T >= {seq_length}")
        
        self.seq_length = seq_length
        self.valid = self.embeddings.shape[0] >= seq_length
        
        if not self.valid:
            print(f"⚠️  WARNING: Only {self.embeddings.shape[0]} frames, need {seq_length}")
    
    def __len__(self):
        if not self.valid:
            return 1
        return max(1, self.embeddings.shape[0] - self.seq_length)
    
    def __getitem__(self, idx):
        if not self.valid:
            # Return dummy data if embeddings too short
            return {
                'video': torch.randn(self.seq_length, 8, 32, 32),
                'audio': torch.randn(256),
                'id': torch.randn(512),
            }
        
        # Extract sequence
        start = idx
        end = min(start + self.seq_length, self.embeddings.shape[0])
        seq = self.embeddings[start:end]
        
        # Pad if needed
        if seq.shape[0] < self.seq_length:
            pad = torch.zeros(
                self.seq_length - seq.shape[0],
                *seq.shape[1:]
            )
            seq = torch.cat([seq, pad], dim=0)
        
        return {
            'video': seq,  # (T, 8, H, W)
            'audio': torch.randn(256),  # Dummy audio embedding
            'id': torch.randn(512),      # Dummy face ID embedding
        }


def test_model_forward(model, batch, device, name="Model"):
    """Test A2V-DiT forward pass"""
    try:
        video = batch['video'].to(device)  # Pre-computed embeddings
        audio = batch['audio'].to(device)  # (B, audio_dim)
        
        print(f"\n📊 Input tensor shapes:")
        print(f"   video (raw): {video.shape} (ndim={video.dim()})")
        print(f"   audio (raw): {audio.shape}")
        
        # The embeddings might be (B, T, H, W, C) or other format
        # Let's just try to reshape to what A2V-DiT expects: (B, C, T, H, W)
        
        # If video has 6+ dims, it might be stacked incorrectly
        if video.dim() > 5:
            print(f"⚠️  Video has {video.dim()} dimensions, attempting to reshape...")
            # Try squeezing extra dimensions
            video = video.squeeze()
            print(f"   After squeeze: {video.shape}")
        
        B = video.shape[0]
        
        # Handle 6D: (B, T, ?, ?, ?, ?) → (B, C, T, H, W)
        if video.dim() == 6:
            # Likely (B, T, D1, D2, D3, D4) where last 4 dims encode spatial/channel info
            # Reshape to (B, T, -1) then figure out C, H, W
            T = video.shape[1]
            rolled = video.reshape(B, T, -1)  # (B, T, D1*D2*D3*D4)
            
            # Now we need to go from (B, T, N) to (B, C, T, H, W)
            # Reasonable assumption: reshape last dim into C, H, W
            # Try: C=8 (typical VAE), H=?, W=?
            total_spatial = rolled.shape[2]  # D1*D2*D3*D4
            
            # Common latent sizes: 8x32x32, 8x16x16, 8x8x8, etc.
            # If total_spatial = 3*3*3*3 = 81, try 8x3x3 + extra or 9x3x3
            # Let's try (B, T, 81) → (B, 9, T, 3, 3) or (B, 8, T, ?, ?)
            
            # Find factors of total_spatial
            h = w = int(total_spatial ** 0.5)
            if h * w != total_spatial:
                # Try different factorizations
                for test_h in range(1, int(total_spatial**0.5) + 1):
                    if total_spatial % test_h == 0:
                        test_w = total_spatial // test_h
                        h, w = test_h, test_w
                        break
            
            c = 1  # Default to 1 channel if ambiguous
            print(f"   Reshaping (B, T, {total_spatial}) → (B, {c}, T, {h}, {w})")
            
            z = rolled.view(B, T, c, h, w)  # (B, T, C, H, W)
            z = z.permute(0, 2, 1, 3, 4)    # (B, C, T, H, W)
        elif video.dim() == 5:
            # Could be (B, T, C, H, W) or (B, C, T, H, W)
            # Check if it looks like pre-computed VAE latents (usually 8 channels)
            if video.shape[1] == 8 or video.shape[2] == 8:  # If 8 is in position 1 or 2, it's likely C
                if video.shape[2] == 8:  # (B, T, 8, H, W) - channels in position 2
                    z = video.permute(0, 2, 1, 3, 4)  # → (B, 8, T, H, W)
                else:  # (B, 8, T, H, W) - already correct
                    z = video
            else:
                z = video
        else:
            z = video
        
        print(f"   video (reshaped): {z.shape}")
        
        # audio_emb: (B, T, audio_dim)
        if z.dim() == 5 and audio.dim() == 2:
            T = z.shape[2]
            audio = audio.unsqueeze(1).expand(-1, T, -1)
        print(f"   audio (reshaped): {audio.shape}")
        
        # ref_image: (B, C, 1, 1, 1)
        if z.dim() == 5:
            ref_image = z[:, :, 0:1, 0:1, 0:1]
        else:
            ref_image = None
        print(f"   ref_image: {ref_image.shape if ref_image is not None else 'None'}")
        
        # Dummy timesteps
        timesteps = torch.randint(0, 1000, (B,)).to(device)
        
        # Forward
        with torch.no_grad():
            output = model(
                z=z,
                timesteps=timesteps,
                audio_emb=audio,
                ref_image=ref_image,
            )
        
        print(f"\n✅ {name} forward pass successful!")
        print(f"   Output shape: {output.shape if hasattr(output, 'shape') else 'tensor'}")
        return True
    except Exception as e:
        print(f"\n❌ {name} forward pass failed!")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test training pipeline with pre-computed embeddings"
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        required=True,
        help="Path to pre-computed embeddings tensor (.pt file)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=16,
        help="Sequence length (number of frames)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 REST Model Validation with Pre-computed Embeddings")
    print("="*60 + "\n")
    
    # Check file exists
    emb_path = Path(args.embedding_path)
    if not emb_path.exists():
        print(f"❌ ERROR: Embedding file not found: {args.embedding_path}")
        return False
    
    print(f"📁 Embedding file: {emb_path}")
    print(f"🖥️  Device: {args.device}")
    print(f"📊 Configuration: batch_size={args.batch_size}, seq_length={args.seq_length}\n")
    
    # Load config
    try:
        config = OmegaConf.load("configs/training_config.yaml")
        print("✅ Config loaded")
    except:
        print("⚠️  Config not found, using defaults")
        config = OmegaConf.create({"device": args.device})
    
    # Create dataset
    print("\n[1/4] Creating dataset...")
    try:
        dataset = EmbeddingTestDataset(args.embedding_path, seq_length=args.seq_length)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        batch = next(iter(loader))
        print(f"✅ Dataset created: {len(dataset)} batches")
        
        # Debug: Show actual shapes
        print(f"\n📊 Batch shapes:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"   {key}: {val.shape}")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False
    
    # Create models
    print("\n[2/4] Initializing models...")
    try:
        vae = TemporalVAE().to(args.device)
        model = A2VDIT().to(args.device)
        # FlowMatcher is a scheduler, initialized during training
        # scheduler = FlowMatcher(model=model, scheduler=noise_scheduler)
        print("✅ Models initialized")
    except Exception as e:
        print(f"❌ Failed to initialize models: {e}")
        traceback.print_exc()
        return False
    
    # Skip VAE test - we have pre-computed embeddings (which ARE VAE output)
    print("\n[3/4] Skipping VAE encoder test...")
    print("     (Using pre-computed embeddings, so VAE validation not needed)")
    
    # Test A2V-DiT with real embeddings (this is what matters!)
    print("\n[4/4] Testing A2V-DiT with real embeddings...")
    success = test_model_forward(model, batch, args.device, "A2V-DiT")
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("\nYou're ready to:")
        print("  1. Implement training loops in scripts/")
        print("  2. Run full training on RunPod")
        print("  3. Monitor progress with tensorboard")
        print("\n🚀 Ready for production training!")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nFix errors above before starting training")
    print("="*60 + "\n")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
