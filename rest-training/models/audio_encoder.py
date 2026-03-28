"""
Audio encoders: SpeechAE using Whisper-tiny
For real-time audio-to-video alignment
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class SpeechAE(nn.Module):
    """
    Speech AutoEncoder using Whisper-tiny for feature extraction
    Compresses audio to compact embeddings aligned with video
    """
    
    def __init__(self, audio_dim: int = 384, output_dim: int = 256):
        super().__init__()
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        
        # Try to use actual Whisper if available, otherwise use dummy
        try:
            import whisper
            self.whisper_model = whisper.load_model("tiny")
            self.use_whisper = True
        except ImportError:
            print("Whisper not available, using dummy audio encoder")
            self.use_whisper = False
        
        # Compression network
        self.encoder = nn.Sequential(
            nn.Linear(audio_dim, audio_dim),
            nn.ReLU(),
            nn.Linear(audio_dim, output_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, audio_dim),
            nn.ReLU(),
            nn.Linear(audio_dim, audio_dim),
        )
    
    def encode_whisper(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features using Whisper-tiny"""
        if not self.use_whisper:
            return self._dummy_encode(audio)
        
        # Convert to numpy, process with whisper
        audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        
        with torch.no_grad():
            mel = self.whisper_model.encoder(torch.from_device(audio_np))
        
        return mel
    
    def _dummy_encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Dummy encoder for testing (without Whisper)"""
        # Placeholder: just return random features of expected shape
        B, audio_len = audio.shape[:2]
        return torch.randn(B, audio_len // 160, self.audio_dim, device=audio.device)
    
    def forward(self, audio: torch.Tensor) -> Tensor:
        """
        audio: (B, audio_len) - raw audio waveform or (B, audio_len, audio_dim) features
        
        Returns: (B, compressed_len, output_dim) - compressed audio embeddings
        """
        if len(audio.shape) == 2:
            # Raw waveform - extract features
            features = self._dummy_encode(audio)  # (B, compressed_len, audio_dim)
        else:
            features = audio  # Already features
        
        # Compress
        compressed = self.encoder(features)  # (B, compressed_len, output_dim)
        
        return compressed
    
    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from compressed"""
        return self.decoder(compressed)


class AudioProcessor(nn.Module):
    """
    Processes audio for streaming generation
    Handles temporal alignment with video chunks
    """
    
    def __init__(self, output_dim: int = 256, chunk_len: int = 4):
        super().__init__()
        self.output_dim = output_dim
        self.chunk_len = chunk_len  # Audio frames per video chunk
        
        self.speech_ae = SpeechAE(output_dim=output_dim)
        
        # Temporal positional encoding
        self.pos_emb = nn.Embedding(1000, output_dim)
    
    def process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, audio_len) - raw waveform
        
        Returns: (B, audio_frames, output_dim)
        """
        # Encode audio
        audio_emb = self.speech_ae(audio)  # (B, audio_frames, output_dim)
        
        # Add positional encoding
        positions = torch.arange(audio_emb.shape[1], device=audio_emb.device)
        pos_emb = self.pos_emb(positions % 1000)  # (audio_frames, output_dim)
        audio_emb = audio_emb + pos_emb.unsqueeze(0)
        
        return audio_emb
    
    def align_audio_to_chunks(
        self,
        audio_emb: torch.Tensor,
        num_chunks: int,
    ) -> torch.Tensor:
        """
        Align audio embeddings to video chunks
        
        audio_emb: (B, audio_frames, output_dim)
        num_chunks: number of video chunks
        
        Returns: (B, num_chunks, chunk_audio_frames, output_dim)
        """
        B, audio_frames, dim = audio_emb.shape
        
        # Calculate frames per chunk
        frames_per_chunk = audio_frames // num_chunks
        
        # Reshape and pad if needed
        if audio_frames % num_chunks != 0:
            # Pad to align
            pad_len = (num_chunks - audio_frames % num_chunks) % num_chunks
            audio_emb = torch.nn.functional.pad(audio_emb, (0, 0, 0, pad_len))
        
        # Reshape into chunks
        audio_chunks = audio_emb.view(B, num_chunks, frames_per_chunk, dim)
        
        return audio_chunks


# For compatibility
from torch import Tensor
