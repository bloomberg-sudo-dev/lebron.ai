#!/usr/bin/env python3
"""
Inference - Load trained model and process frames/videos
"""

import os
import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import json


# Import model (same as training)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn as nn


class SimpleVideoModel(nn.Module):
    """Simple CNN for video frames (same as training)"""
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


def load_model(checkpoint_path, device):
    """Load trained model"""
    model = SimpleVideoModel().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    return model


def process_frame(model, frame_path, device):
    """Process single frame"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(frame_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(img)
    
    return embedding.cpu().numpy()


def process_directory(model, dataset_dir, output_file, device, max_frames=None):
    """Process all frames in directory"""
    frames = sorted(Path(dataset_dir).glob("*/frame_*.jpg"))
    
    if max_frames:
        frames = frames[:max_frames]
    
    print(f"📁 Processing {len(frames)} frames...")
    
    results = {}
    
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        for i, frame_path in enumerate(frames):
            img = Image.open(frame_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            
            embedding = model(img)
            
            results[str(frame_path)] = embedding.cpu().numpy().tolist()
            
            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(frames)}] Processed")
    
    # Save results
    print(f"💾 Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Embeddings saved: {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (model_final.pt)")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory with frames")
    parser.add_argument("--output-file", type=str, default="embeddings.json", help="Output JSON with embeddings")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}\n")
    
    # Load model
    model = load_model(args.model, device)
    
    # Process frames
    results = process_directory(model, args.dataset_dir, args.output_file, device, args.max_frames)
    
    print(f"\n✅ Done!")
    print(f"📊 Processed: {len(results)} frames")
    print(f"📁 Embeddings: {args.output_file}")


if __name__ == "__main__":
    main()
