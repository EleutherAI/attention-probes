import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Optional


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, n_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, n_heads, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim) with interleaved pattern
        sin: Sine embeddings of shape (seq_len, head_dim) with interleaved pattern
        position_ids: Position IDs of shape (batch_size, seq_len)
    
    Returns:
        q_rotated, k_rotated: Rotated query and key tensors
    """
    # Ensure position_ids is on the same device as cos/sin
    if position_ids.device != cos.device:
        position_ids = position_ids.to(cos.device)
    
    # Get embeddings for the specific positions
    cos_emb = cos[position_ids]  # (batch_size, seq_len, head_dim)
    sin_emb = sin[position_ids]  # (batch_size, seq_len, head_dim)
    
    # Expand to match the head dimension and broadcast to all heads
    cos_emb = cos_emb.unsqueeze(2).expand(-1, -1, q.shape[2], -1)  # (batch_size, seq_len, n_heads, head_dim)
    sin_emb = sin_emb.unsqueeze(2).expand(-1, -1, q.shape[2], -1)  # (batch_size, seq_len, n_heads, head_dim)
    
    # Apply rotary embeddings using the interleaved pattern
    # The embeddings are already in the correct pattern: [cos_0, sin_0, cos_1, sin_1, ...]
    q_rotated = q * cos_emb - torch.roll(q, shifts=1, dims=-1) * sin_emb
    k_rotated = k * cos_emb - torch.roll(k, shifts=1, dims=-1) * sin_emb
    
    return q_rotated, k_rotated


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize RoPE embeddings
        self.register_buffer('cos_emb', None)
        self.register_buffer('sin_emb', None)
        
    def _init_rope_embeddings(self, max_seq_len, device=None):
        """Initialize RoPE embeddings."""
        position = torch.arange(0, max_seq_len, dtype=torch.float, device=device)
        
        # Create frequency bands for RoPE
        freqs = torch.exp(
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) * 
            (-math.log(10000.0) / self.head_dim)
        )
        
        # Create angles for each position and frequency
        angles = position.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, head_dim//2)
        
        # Create cos and sin embeddings
        cos_emb = torch.cos(angles)  # (seq_len, head_dim//2)
        sin_emb = torch.sin(angles)  # (seq_len, head_dim//2)
        
        # Interleave cos and sin to create full head_dim embeddings
        # This creates the pattern: [cos_0, sin_0, cos_1, sin_1, ...]
        self.cos_emb = torch.zeros(max_seq_len, self.head_dim, device=device)
        self.sin_emb = torch.zeros(max_seq_len, self.head_dim, device=device)
        
        self.cos_emb[:, 0::2] = cos_emb  # Even indices get cos
        self.sin_emb[:, 1::2] = sin_emb  # Odd indices get sin
    
    def forward(self, x, mask=None, position_ids=None):
        batch_size, seq_len, d_model = x.shape
        
        # Initialize RoPE embeddings if needed
        if self.cos_emb is None or self.cos_emb.size(0) < seq_len:
            self._init_rope_embeddings(seq_len, device=x.device)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        q, k = apply_rotary_pos_emb(q, k, self.cos_emb, self.sin_emb, position_ids)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.o_proj(context)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with ReLU² activation."""
    
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout
    
    def forward(self, x):
        # ReLU² activation: (ReLU(x))²
        return self.w2(F.dropout(F.relu(self.w1(x))**2, p=self.dropout, training=self.training))


class TransformerBlock(nn.Module):
    """Transformer block with layer normalization."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x, mask=None, position_ids=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask, position_ids)
        x = x + F.dropout(attn_output, p=self.dropout, training=self.training)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + F.dropout(ff_output, p=self.dropout, training=self.training)
        x = self.norm2(x)
        
        return x


class SkylineProbe(nn.Module):
    """
    Skyline probe: a transformer with RoPE and ReLU² activation.
    Compatible with the AttentionProbe interface.
    """
    
    def __init__(self, d_in, n_heads, output_dim=1, config=None):
        super().__init__()
        
        self.d_model = d_in
        self.d_ff = 2 * self.d_model  # 2x width for MLP as requested
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.n_layers = 3  # 3 layers as requested
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, n_heads, self.d_ff, dropout=0.0)
            for _ in range(self.n_layers)
        ])
        
        # Output projection (direct to output_dim)
        self.output_proj = nn.Linear(self.d_model, output_dim)
        
        # For compatibility with AttentionProbe interface
        self.config = config
        
        # Dummy attributes for compatibility
        self.q = nn.Linear(d_in, n_heads, bias=False)  # Dummy for compatibility
        self.v = nn.Linear(d_in, n_heads * output_dim)  # Dummy for compatibility
        self.position_weight = nn.Parameter(torch.zeros((n_heads,), dtype=torch.float32))  # Dummy
        self.attn_hook = nn.Identity()  # Dummy hook for compatibility
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask, position):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            mask: Mask tensor of shape (batch_size, seq_len)
            position: Position tensor of shape (batch_size, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask, position)
        
        # Global pooling: take mean over sequence length (masked)
        if mask is not None:
            # Apply mask and compute mean
            masked_x = x * mask.unsqueeze(-1)
            seq_lengths = mask.sum(dim=-1, keepdim=True)  # (batch_size, 1)
            seq_lengths = torch.clamp(seq_lengths, min=1)
            # Sum over sequence dimension and divide by sequence lengths
            pooled = masked_x.sum(dim=1) / seq_lengths  # (batch_size, d_model)
        else:
            pooled = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_proj(pooled)
        
        return output 