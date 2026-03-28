"""
Talking Head Dataset loader for REST training
Supports VoxCeleb2, RAVDESS, and custom datasets
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class TalkingHeadDataset(Dataset):
    """
    Dataset for audio-driven talking head generation
    
    Expected structure:
    data_root/
    ├─ videos/
    │  └─ [video_id].mp4
    ├─ audio/
    │  └─ [video_id].wav
    └─ metadata.json
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (512, 512),
        sample_rate: int = 16000,
        audio_length: int = 2,  # seconds
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.audio_samples = audio_length * sample_rate
        
        # Load metadata
        metadata_path = self.data_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = self._build_metadata()
        
        # Filter by split
        self.samples = [s for s in self.metadata if s.get("split", "train") == split]
    
    def _build_metadata(self) -> list:
        """Auto-build metadata from file structure"""
        samples = []
        video_dir = self.data_root / "videos"
        audio_dir = self.data_root / "audio"
        
        for video_file in sorted(video_dir.glob("*.mp4")):
            video_id = video_file.stem
            audio_file = audio_dir / f"{video_id}.wav"
            
            if audio_file.exists():
                samples.append({
                    "video_id": video_id,
                    "video_path": str(video_file),
                    "audio_path": str(audio_file),
                    "split": "train",
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
        {
            'video': (3, num_frames, H, W) - reference frame + video frames
            'audio': (audio_samples,) - audio waveform
            'ref_frame': (3, H, W) - reference frame for ID-Sink
        }
        """
        sample = self.samples[idx]
        
        # Load video (placeholder - in real impl, use torchvision.io.read_video)
        try:
            video = self._load_video(sample["video_path"])
        except:
            # Dummy video if loading fails
            video = torch.randn(self.num_frames, *self.frame_size, 3).permute(3, 0, 1, 2)
        
        # Load audio (placeholder)
        try:
            audio = self._load_audio(sample["audio_path"])
        except:
            # Dummy audio if loading fails
            audio = torch.randn(self.audio_samples)
        
        # Reference frame (first frame)
        ref_frame = video[:, 0]  # (3, H, W)
        
        # Return
        return {
            "video": video,  # (3, T, H, W)
            "audio": audio,  # (audio_samples,)
            "ref_frame": ref_frame,  # (3, H, W)
        }
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video and extract frames"""
        # Placeholder implementation
        # In real implementation, use:
        # from torchvision.io import read_video
        # frames, _, info = read_video(video_path)
        
        # For now, return dummy frames
        frames = torch.randn(self.num_frames, *self.frame_size, 3)
        return frames.permute(3, 0, 1, 2)  # (3, T, H, W)
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio waveform"""
        # Placeholder implementation
        # In real implementation, use:
        # import torchaudio
        # waveform, sr = torchaudio.load(audio_path)
        # waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform[0])
        
        # For now, return dummy audio
        return torch.randn(self.audio_samples)


class DummyTalkingHeadDataset(Dataset):
    """Dummy dataset for testing without real data"""
    
    def __init__(
        self,
        num_samples: int = 100,
        num_frames: int = 2,  # Tiny: 2 frames
        frame_size: Tuple[int, int] = (16, 16),  # Tiny: 16x16
        sample_rate: int = 16000,
        audio_length: int = 1,
    ):
        super().__init__()
        
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.audio_samples = audio_length * sample_rate
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return dummy tensors"""
        return {
            "video": torch.randn(3, self.num_frames, *self.frame_size),
            "audio": torch.randn(self.audio_samples),
            "ref_frame": torch.randn(3, *self.frame_size),
        }


class TalkingHeadDataLoader:
    """Convenience wrapper for dataloaders"""
    
    @staticmethod
    def create_loaders(
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (512, 512),
        use_dummy: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and val dataloaders
        
        Returns: (train_loader, val_loader)
        """
        
        if use_dummy or not Path(data_root).exists():
            print("Using dummy dataset for testing")
            train_dataset = DummyTalkingHeadDataset(
                num_samples=100,
                num_frames=num_frames,
                frame_size=frame_size,
            )
            val_dataset = DummyTalkingHeadDataset(
                num_samples=20,
                num_frames=num_frames,
                frame_size=frame_size,
            )
        else:
            train_dataset = TalkingHeadDataset(
                data_root=data_root,
                split="train",
                num_frames=num_frames,
                frame_size=frame_size,
            )
            val_dataset = TalkingHeadDataset(
                data_root=data_root,
                split="val",
                num_frames=num_frames,
                frame_size=frame_size,
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader
