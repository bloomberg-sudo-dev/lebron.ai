"""
Simplified Temporal VAE for video compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SimpleTemporalVAE(nn.Module):
    """
    Simple VAE: (B, 3, T, H, W) -> latent -> (B, 3, T, H, W)
    No fancy FC layers, just straightforward conv-based compression
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
            hidden_dims = [64, 128, 256]
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.kl_weight = kl_weight
        
        # Encoder: (B, 3, T, H, W) -> (B, latent_channels, T, H, W)
        encoder_layers = [
            nn.Conv3d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dims[2], latent_channels * 2, kernel_size=3, padding=1),  # *2 for mu and logvar
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: (B, latent_channels, T, H, W) -> (B, 3, T, H, W)
        decoder_layers = [
            nn.Conv3d(latent_channels, hidden_dims[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dims[2], hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dims[1], hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dims[0], in_channels, kernel_size=3, padding=1),
        ]
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to latent distribution
        Returns: z_sampled, mu, logvar
        """
        # Encode and split into mu, logvar
        encoded = self.encoder(x)
        mu, logvar = encoded.chunk(2, dim=1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, T, H, W) video
        
        Returns:
            recon: (B, 3, T, H, W) reconstructed video
            mu: (B, latent_channels, T, H, W) mean
            logvar: (B, latent_channels, T, H, W) log variance
        """
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def vae_loss(self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss: reconstruction + KL divergence
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Weighted sum
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


# Aliases for backward compatibility
TemporalVAE = SimpleTemporalVAE
TemporalVAEEncoder = SimpleTemporalVAE
TemporalVAEDecoder = SimpleTemporalVAE
