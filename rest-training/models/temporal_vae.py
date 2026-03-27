"""
Temporal VAE for compact video latent space (32x32x8 compression)
Based on LTX-Video architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ResBlock3D(nn.Module):
    """3D Residual Block for temporal compression"""
    
    def __init__(self, channels: int, kernel_size: int = 3, use_temporal: bool = True):
        super().__init__()
        self.use_temporal = use_temporal
        pad = kernel_size // 2
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(1, kernel_size, kernel_size), padding=(0, pad, pad))
        
        if use_temporal:
            self.norm_t = nn.GroupNorm(32, channels)
            self.conv_t = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1), padding=(pad, 0, 0))
        
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(1, kernel_size, kernel_size), padding=(0, pad, pad))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W)"""
        residual = x
        
        # Spatial convolution
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Temporal convolution
        if self.use_temporal:
            x = self.norm_t(x)
            x = F.silu(x)
            x = self.conv_t(x)
        
        # Final spatial
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + residual


class TemporalVAEEncoder(nn.Module):
    """
    Encodes video (H, W, F) -> compact latent (h, w, f) with 32x32x8 compression
    
    Spatial: H, W -> H/32, W/32
    Temporal: F -> F/8
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # Initial conv
        self.conv_in = nn.Conv3d(in_channels, hidden_dims[0], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Downsampling blocks: spatial 2x2x2 = 8x total, temporal 2x = temporal reduction
        layers = []
        in_ch = hidden_dims[0]
        
        for i, out_ch in enumerate(hidden_dims[1:]):
            # Spatial downsample 2x (total 2->4->8->16x)
            layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)))
            layers.append(ResBlock3D(out_ch, use_temporal=(i < 2)))  # Temporal in first 2 blocks
            
            # Temporal downsample 2x in first block
            if i == 0:
                layers.append(nn.Conv3d(out_ch, out_ch, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0))
            
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # Output projection to latent
        self.norm = nn.GroupNorm(32, hidden_dims[-1])
        self.conv_out = nn.Conv3d(hidden_dims[-1], latent_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) - video frames
        Returns: (B, latent_channels, T/8, H/32, W/32)
        """
        x = self.conv_in(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.conv_out(x)
        return x


class TemporalVAEDecoder(nn.Module):
    """
    Decodes compact latent -> video (H, W, F)
    Inverse of encoder: (h, w, f) -> (H, W, F) with 32x32x8 compression
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        
        # Initial projection
        self.conv_in = nn.Conv3d(latent_channels, hidden_dims[0], kernel_size=1)
        
        # Upsampling blocks
        layers = []
        in_ch = hidden_dims[0]
        
        for i, out_ch in enumerate(hidden_dims[1:]):
            # Temporal upsample 2x in first block
            if i == 0:
                layers.append(nn.ConvTranspose3d(in_ch, in_ch, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0))
            
            # Spatial upsample 2x
            layers.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)))
            layers.append(ResBlock3D(out_ch, use_temporal=(i < 1)))
            
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
        
        # Output
        self.norm = nn.GroupNorm(32, hidden_dims[-1])
        self.conv_out = nn.Conv3d(hidden_dims[-1], out_channels, kernel_size=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_channels, T/8, H/32, W/32)
        Returns: (B, out_channels, T, H, W)
        """
        x = self.conv_in(z)
        x = self.decoder(x)
        x = self.norm(x)
        x = self.conv_out(x)
        return x


class TemporalVAE(nn.Module):
    """
    Complete VAE for video compression
    Compresses: (B, 3, T, H, W) -> (B, 4, T/8, H/32, W/32)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_dims: List[int] = None,
        kl_weight: float = 0.00001,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        self.encoder = TemporalVAEEncoder(in_channels, latent_channels, hidden_dims)
        self.decoder = TemporalVAEDecoder(latent_channels, in_channels, hidden_dims[::-1])
        
        # KL divergence weight (beta-VAE)
        self.kl_weight = kl_weight
        self.latent_channels = latent_channels
        
        # Distribution parameters (mu, logvar)
        self.fc_mu = nn.Linear(latent_channels, latent_channels)
        self.fc_logvar = nn.Linear(latent_channels, latent_channels)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to latent distribution"""
        z = self.encoder(x)
        # Flatten spatial dims for distribution
        B, C, T, H, W = z.shape
        z_flat = z.view(B, C, -1).mean(dim=2, keepdim=True)  # (B, C, 1)
        
        mu = self.fc_mu(z_flat.squeeze(-1).T).T  # (B, C)
        logvar = self.fc_logvar(z_flat.squeeze(-1).T).T  # (B, C)
        
        return z, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode -> reparameterize -> decode
        
        Returns: (reconstruction, mu, logvar)
        """
        z_encoded, mu, logvar = self.encode(x)
        z_sampled = self.reparameterize(mu, logvar)
        
        # Restore shape for decoder
        B, C = z_sampled.shape
        z_reconstructed = z_encoded.clone()
        
        recon_x = self.decode(z_reconstructed)
        
        return recon_x, mu, logvar
    
    def loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """VAE loss = reconstruction + KL divergence"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
        }
