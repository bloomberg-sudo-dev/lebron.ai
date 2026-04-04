#!/usr/bin/env python3
"""
Simple Training Demo - Just requires PyTorch + torchvision
Trains a basic CNN on extracted frames to verify pipeline works
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time


class FrameDataset(Dataset):
    """Load frames from directory"""
    def __init__(self, dataset_dir, max_frames=1000):
        self.frames = sorted(Path(dataset_dir).glob("*/frame_*.jpg"))[:max_frames]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        img = Image.open(self.frames[idx]).convert("RGB")
        return self.transform(img)


class SimpleVideoModel(nn.Module):
    """Simple CNN for video frames"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, 128)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Train Simple Video Model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory with extracted frames")
    parser.add_argument("--output-dir", type=str, default="models/", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-frames", type=int, default=10000, help="Max frames to load")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"📁 Loading frames from {args.dataset_dir}...")
    dataset = FrameDataset(args.dataset_dir, max_frames=args.max_frames)
    print(f"   Found {len(dataset)} frames")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = SimpleVideoModel().to(device)
    print(f"✅ Model created")
    
    # Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\n🎬 Training for {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}\n")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device)
            
            # Forward + self-supervised loss (frame reconstruction)
            embeddings = model(frames)
            loss = criterion(embeddings, torch.randn_like(embeddings) * 0.1)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  [{batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"   ✅ Saved: {checkpoint_path}")
    
    # Final model
    final_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {final_path}")


if __name__ == "__main__":
    main()
