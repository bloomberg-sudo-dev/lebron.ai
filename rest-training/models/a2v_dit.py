"""
A2V-DiT: Audio-to-Video Diffusion Transformer with ID-Context Cache
28 transformer blocks with streaming support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from .id_context_cache import IDContextCache


class TimestepEmbedding(nn.Module):
    """Embedding for diffusion timestep"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.linear = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B,) or (B, 1)
        Returns: (B, dim)
        """
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            * (torch.log(torch.tensor(self.max_period, dtype=torch.float32)) / half_dim)
        )
        args = timesteps.float() * freqs
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.linear(embedding)


class AudioCrossAttention(nn.Module):
    """Frame-level 2D cross-attention for audio conditioning"""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        audio_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, seq_len, hidden_dim) - video features
        audio_cond: (B, audio_len, hidden_dim) - audio embeddings
        
        Returns: (B, seq_len, hidden_dim)
        """
        B, seq_len, _ = x.shape
        
        Q = self.q_proj(x)  # (B, seq_len, hidden_dim)
        K = self.k_proj(audio_cond)  # (B, audio_len, hidden_dim)
        V = self.v_proj(audio_cond)
        
        # Reshape for multi-head
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, audio_cond.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, audio_cond.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        
        return out


class DiTBlock(nn.Module):
    """Single DiT block with ID-Context Cache + Audio conditioning"""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12):
        super().__init__()
        
        # ID-Context Cache self-attention
        self.id_context_cache = IDContextCache(hidden_dim, num_heads)
        
        # Audio cross-attention
        self.audio_cross_attn = AudioCrossAttention(hidden_dim, num_heads)
        self.norm_audio = nn.LayerNorm(hidden_dim)
        
        # Timestep conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_time = nn.LayerNorm(hidden_dim)
        
        # Output FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        audio_cond: Optional[torch.Tensor] = None,
        id_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, seq_len, hidden_dim) - latent features
        timestep_emb: (B, hidden_dim) - timestep embedding
        audio_cond: (B, audio_len, hidden_dim) - audio conditioning
        id_anchor: (B, 1, hidden_dim) - ID reference
        
        Returns: (B, seq_len, hidden_dim)
        """
        # 1. ID-Context Cache self-attention
        x_attn = self.id_context_cache(x, id_anchor=id_anchor)
        x = x + x_attn
        
        # 2. Audio cross-attention (if provided)
        if audio_cond is not None:
            x_audio = self.audio_cross_attn(x, audio_cond)
            x_audio = self.norm_audio(x_audio)
            x = x + x_audio
        
        # 3. Timestep conditioning
        time_cond = self.time_mlp(timestep_emb).unsqueeze(1)  # (B, 1, hidden_dim)
        x_time = x + time_cond  # Broadcast
        x_time = self.norm_time(x_time)
        
        # 4. FFN
        x_ffn = self.ffn(x_time)
        x = x_time + x_ffn
        x = self.norm_ffn(x)
        
        return x


class A2VDIT(nn.Module):
    """
    Audio-to-Video Diffusion Transformer (28 blocks)
    Streaming-capable with ID-Context Cache
    """
    
    def __init__(
        self,
        latent_dim: int = 4,  # From Temporal VAE output
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 28,
        audio_dim: int = 256,  # SpeechAE output
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Input projection (flatten spatial dims: h*w)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Timestep embedding
        self.timestep_emb = TimestepEmbedding(hidden_dim)
        
        # Audio projection
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
    
    def forward(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        audio_emb: Optional[torch.Tensor] = None,
        ref_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        z: (B, latent_dim, T, H, W) - noisy latents from VAE
        timesteps: (B,) - diffusion timesteps
        audio_emb: (B, T, audio_dim) - audio embeddings (SpeechAE)
        ref_image: (B, latent_dim, 1, 1, 1) - reference frame (for ID-Sink)
        
        Returns: (B, latent_dim, T, H, W) - denoised latents
        """
        B, C, T, H, W = z.shape
        
        # Flatten spatial dims
        z_flat = z.view(B, C, T * H * W).transpose(1, 2)  # (B, T*H*W, C)
        
        # Project latents to hidden dim
        x = self.input_proj(z_flat)  # (B, T*H*W, hidden_dim)
        
        # Timestep embedding
        t_emb = self.timestep_emb(timesteps)  # (B, hidden_dim)
        
        # Audio conditioning
        id_anchor = None
        if ref_image is not None:
            ref_flat = ref_image.view(B, C, -1).transpose(1, 2)  # (B, 1, C)
            id_anchor = self.input_proj(ref_flat)  # (B, 1, hidden_dim)
        
        audio_cond = None
        if audio_emb is not None:
            # Repeat audio to match sequence length
            audio_flat = audio_emb.view(B, -1, audio_emb.shape[-1])  # (B, T_audio, audio_dim)
            audio_cond = self.audio_proj(audio_flat)  # (B, T_audio, hidden_dim)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(
                x,
                timestep_emb=t_emb,
                audio_cond=audio_cond,
                id_anchor=id_anchor,
            )
        
        # Output projection
        x = self.norm_out(x)
        x = self.out_proj(x)  # (B, T*H*W, latent_dim)
        
        # Reshape back
        x = x.transpose(1, 2).view(B, C, T, H, W)
        
        return x
