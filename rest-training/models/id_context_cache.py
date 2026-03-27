"""
ID-Context Cache: Maintains identity + temporal consistency in streaming diffusion
Combines ID-Sink and Context-Cache principles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class IDSink(nn.Module):
    """
    ID-Sink: Maintains reference frame embeddings as persistent anchors
    Preserves identity across all chunks
    """
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer('id_kv_cache', None)
    
    def set_reference(self, ref_embedding: torch.Tensor):
        """Set reference frame (e.g., from I_ref)"""
        # ref_embedding: (B, seq_len, hidden_dim) - usually (B, 1, hidden_dim)
        self.id_kv_cache = ref_embedding
    
    def get_id_anchor(self) -> torch.Tensor:
        """Returns persistent ID anchor"""
        if self.id_kv_cache is None:
            raise ValueError("ID reference not set. Call set_reference() first.")
        return self.id_kv_cache
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ID-conditioned embedding"""
        if self.id_kv_cache is None:
            return x
        
        # Concatenate ID anchor with input
        return torch.cat([self.id_kv_cache, x], dim=1)


class ContextCache(nn.Module):
    """
    Context-Cache: Maintains temporal flow via KV caching
    Concatenates previous chunk KV with current chunk KV
    """
    
    def __init__(self, max_chunks: int = 100):
        super().__init__()
        self.max_chunks = max_chunks
        self.kv_history = []
    
    def push(self, k: torch.Tensor, v: torch.Tensor):
        """Store KV from current chunk"""
        self.kv_history.append((k.detach(), v.detach()))
        
        # Keep only recent history to avoid memory explosion
        if len(self.kv_history) > self.max_chunks:
            self.kv_history.pop(0)
    
    def get_context(self, lookback: int = 1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get KV from previous chunks (default: last chunk)"""
        if len(self.kv_history) == 0:
            return None, None
        
        # Get last 'lookback' chunks
        k_context = []
        v_context = []
        
        for i in range(max(0, len(self.kv_history) - lookback), len(self.kv_history) - 1):
            k_context.append(self.kv_history[i][0])
            v_context.append(self.kv_history[i][1])
        
        if not k_context:
            return None, None
        
        k_context = torch.cat(k_context, dim=1)  # Concatenate along sequence dim
        v_context = torch.cat(v_context, dim=1)
        
        return k_context, v_context
    
    def reset(self):
        """Clear history"""
        self.kv_history = []


class IDContextAttention(nn.Module):
    """
    Attention with ID-Context Cache
    
    Self-attention that uses:
    - ID-Sink: reference frame as persistent anchor
    - Context-Cache: previous chunk KV for temporal continuity
    """
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.id_sink = IDSink(hidden_dim)
        self.context_cache = ContextCache()
    
    def reshape_for_attention(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Reshape (B, seq_len, hidden_dim) -> (B, num_heads, seq_len, head_dim)"""
        B = x.shape[0]
        x = x.view(B, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        id_anchor: Optional[torch.Tensor] = None,
        use_context_cache: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, seq_len, hidden_dim) - current chunk
        id_anchor: (B, 1, hidden_dim) - reference frame embedding
        
        Returns: (B, seq_len, hidden_dim) - attention output
        """
        B, seq_len, _ = x.shape
        
        # Project Q, K, V
        Q = self.q_proj(x)  # (B, seq_len, hidden_dim)
        K_curr = self.k_proj(x)
        V_curr = self.v_proj(x)
        
        # Build key-value sequence using ID-Sink and Context-Cache
        K_list = []
        V_list = []
        
        # 1. Add ID anchor (constant across all chunks)
        if id_anchor is not None:
            K_id = self.k_proj(id_anchor)  # (B, 1, hidden_dim)
            V_id = self.v_proj(id_anchor)
            K_list.append(K_id)
            V_list.append(V_id)
        
        # 2. Add context from previous chunks
        if use_context_cache:
            k_context, v_context = self.context_cache.get_context(lookback=1)
            if k_context is not None:
                K_list.append(k_context)  # (B, ctx_len, hidden_dim)
                V_list.append(v_context)
        
        # 3. Add current chunk
        K_list.append(K_curr)
        V_list.append(V_curr)
        
        # Concatenate all K, V
        K = torch.cat(K_list, dim=1)  # (B, total_seq, hidden_dim)
        V = torch.cat(V_list, dim=1)
        
        # Reshape for multi-head attention
        Q = self.reshape_for_attention(Q, seq_len)  # (B, num_heads, seq_len, head_dim)
        K = self.reshape_for_attention(K, K.shape[1])
        V = self.reshape_for_attention(V, V.shape[1])
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_len, total_seq)
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, seq_len, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Cache current K, V for next chunk
        self.context_cache.push(K_curr, V_curr)
        
        return output
    
    def set_id_reference(self, ref_embedding: torch.Tensor):
        """Set reference frame for ID-Sink"""
        self.id_sink.set_reference(ref_embedding)
    
    def reset_cache(self):
        """Reset context cache (start new sequence)"""
        self.context_cache.reset()


class IDContextCache(nn.Module):
    """
    Complete ID-Context Cache module combining ID-Sink and Context-Cache
    For use in A2V-DiT transformer blocks
    """
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.attention = IDContextAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        id_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, seq_len, hidden_dim)
        id_anchor: (B, 1, hidden_dim) - reference frame
        
        Returns: (B, seq_len, hidden_dim)
        """
        # Self-attention with ID-Context Cache
        attn_out = self.attention(x, id_anchor=id_anchor, use_context_cache=True)
        x = x + attn_out  # Residual
        x = self.norm(x)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = x + ffn_out  # Residual
        x = self.norm_ffn(x)
        
        return x
    
    def set_id_reference(self, ref_embedding: torch.Tensor):
        """Set reference frame"""
        self.attention.set_id_reference(ref_embedding)
    
    def reset_cache(self):
        """Reset cache for new sequence"""
        self.attention.reset_cache()
