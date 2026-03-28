#!/usr/bin/env python
"""
Simple inference: audio → latents → video frames
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TemporalVAE, SpeechAE
from scripts.train_teacher_simple import SimpleTeacher


def inference(audio_path, output_dir="outputs"):
    """Generate video frames from audio"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load models
    print("Loading models...")
    vae = TemporalVAE(in_channels=3, latent_channels=4, hidden_dims=[128, 256, 512]).to(device)
    vae.load_state_dict(torch.load("checkpoints/vae_best.pt", map_location=device))
    vae.eval()
    
    teacher = SimpleTeacher(audio_dim=256, latent_dim=4, seq_len=512).to(device)
    teacher.load_state_dict(torch.load("checkpoints/teacher_simple.pt", map_location=device))
    teacher.eval()
    
    audio_encoder = SpeechAE(audio_dim=384, output_dim=256).to(device)
    audio_encoder.eval()
    
    print("✅ Models loaded")
    
    # Load audio (dummy for now)
    print(f"Loading audio: {audio_path}")
    audio = torch.randn(1, 16000).to(device)  # 1 second at 16kHz
    print(f"Audio shape: {audio.shape}")
    
    # Encode audio
    with torch.no_grad():
        audio_emb = audio_encoder(audio)
        print(f"Audio embedding shape: {audio_emb.shape}")
        
        # Generate latents
        latents = teacher(audio_emb)
        print(f"Generated latents shape: {latents.shape}")
        
        # Reshape for VAE decoder
        latents_reshaped = latents.view(1, 4, 2, 16, 16)  # (B, C, T, H, W)
        print(f"Latents reshaped: {latents_reshaped.shape}")
        
        # Decode to video
        video = vae.decode(latents_reshaped)
        print(f"Generated video shape: {video.shape}")
        
        # Save frames
        for i in range(video.shape[2]):
            frame = video[0, :, i].permute(1, 2, 0).cpu().numpy()
            frame = ((frame + 1) / 2 * 255).astype('uint8')  # Denormalize
            
            # Save as image
            import cv2
            cv2.imwrite(str(output_dir / f"frame_{i:04d}.png"), frame)
    
    print(f"✅ Generated {video.shape[2]} frames in {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="dummy", help="Audio path (or 'dummy' for test)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()
    
    inference(args.audio, args.output)
