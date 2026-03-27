"""
Flow Matching scheduler for faster diffusion convergence
Alternative to DDPM with fewer steps needed
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class FlowMatchingScheduler(nn.Module):
    """
    Flow Matching noise scheduler
    Learns continuous vector field from noise to data distribution
    """
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        super().__init__()
        
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # Create noise schedule
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            betas = self._cosine_schedule(num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Precalculated values for efficiency
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
    
    def _cosine_schedule(self, steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine annealing schedule"""
        t = torch.arange(steps + 1)
        f_t = torch.cos(((t / steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        alphas = f_t / f_t[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to data at timestep t
        
        x0: clean data (B, C, T, H, W)
        t: timestep (B,) with values in [0, num_steps)
        noise: (B, C, T, H, W) optional noise
        
        Returns: (noisy_x, noise) - noisy data and noise used
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas.shape) < len(x0.shape):
            sqrt_alphas = sqrt_alphas.unsqueeze(-1)
            sqrt_one_minus_alphas = sqrt_one_minus_alphas.unsqueeze(-1)
        
        # Forward process: q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps
        noisy_x = sqrt_alphas * x0 + sqrt_one_minus_alphas * noise
        
        return noisy_x, noise
    
    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alphas = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t]
        
        while len(sqrt_alphas.shape) < len(x_t.shape):
            sqrt_alphas = sqrt_alphas.unsqueeze(-1)
            sqrt_one_minus_alphas = sqrt_one_minus_alphas.unsqueeze(-1)
        
        # x_0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
        x0_pred = (x_t - sqrt_one_minus_alphas * noise_pred) / sqrt_alphas
        return x0_pred
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training"""
        return torch.randint(0, self.num_steps, (batch_size,), device=device)
    
    def get_timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Get positional embedding for timestep
        Similar to Transformer positional encoding
        """
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        
        t_emb = torch.einsum("b,f->bf", t.float(), freqs)
        t_emb = torch.cat([torch.cos(t_emb), torch.sin(t_emb)], dim=-1)
        
        if dim % 2 == 1:
            t_emb = torch.nn.functional.pad(t_emb, (0, 1))
        
        return t_emb


class FlowMatcher(nn.Module):
    """
    Flow Matching trainer
    Learns to predict velocity field v(x, t) for ODE: dx/dt = v(x, t)
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: FlowMatchingScheduler,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        
        self.model = model
        self.scheduler = scheduler
        self.lr = learning_rate
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def training_step(
        self,
        x0: torch.Tensor,  # Clean data
        audio_emb: torch.Tensor,  # Audio conditioning
        ref_image: torch.Tensor,  # Reference for ID-Sink
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single training step with Flow Matching objective
        
        Returns: (loss, metrics_dict)
        """
        # Sample random timesteps
        t = self.scheduler.sample_timesteps(x0.shape[0], x0.device)
        
        # Add noise
        xt, noise = self.scheduler.add_noise(x0, t)
        
        # Model prediction
        noise_pred = self.model(
            xt,
            timesteps=t,
            audio_emb=audio_emb,
            ref_image=ref_image,
        )
        
        # MSE loss between predicted and actual noise
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        metrics = {
            "loss": loss.item(),
            "lr": self.lr,
        }
        
        return loss, metrics
    
    def get_timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Helper to get timestep embedding"""
        return self.scheduler.get_timestep_embedding(t, dim)


class AsynchronousNoiseScheduler(nn.Module):
    """
    Asynchronous noise scheduler for Streaming Distillation
    Applies different noise schedules to different chunks
    """
    
    def __init__(self, scheduler: FlowMatchingScheduler):
        super().__init__()
        self.scheduler = scheduler
    
    def add_noise_asynchronous(
        self,
        x0: torch.Tensor,  # Full sequence
        chunk_indices: list,  # Which chunks to add noise to
        chunk_timesteps: list,  # Different timesteps per chunk
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise asynchronously (different noise schedule per chunk)
        Used for teacher training in ASD
        
        x0: (B, C, T, H, W)
        chunk_indices: list of slice objects for each chunk
        chunk_timesteps: list of (B,) timesteps for each chunk
        
        Returns: (noisy_x, timesteps_tensor)
        """
        noisy_x = x0.clone()
        noise = torch.randn_like(x0)
        timesteps_list = []
        
        for idx, (chunk_idx, t_chunk) in enumerate(zip(chunk_indices, chunk_timesteps)):
            # Add noise to this chunk with its own schedule
            sqrt_alphas = self.scheduler.sqrt_alphas_cumprod[t_chunk]
            sqrt_one_minus_alphas = self.scheduler.sqrt_one_minus_alphas_cumprod[t_chunk]
            
            # Reshape for broadcasting
            for _ in range(len(x0.shape) - 1):
                sqrt_alphas = sqrt_alphas.unsqueeze(-1)
                sqrt_one_minus_alphas = sqrt_one_minus_alphas.unsqueeze(-1)
            
            # Add noise
            noisy_x[chunk_idx] = sqrt_alphas * x0[chunk_idx] + sqrt_one_minus_alphas * noise[chunk_idx]
            timesteps_list.append(t_chunk)
        
        timesteps = torch.cat(timesteps_list, dim=0)
        
        return noisy_x, timesteps
