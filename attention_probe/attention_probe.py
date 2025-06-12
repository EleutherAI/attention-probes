from torch import nn
import torch
from typing import Any

class AttentionProbe(nn.Module):
    def __init__(self, d_in, n_heads, output_dim: int = 1, hidden_dim: int = 0, use_tanh: bool = False, config: Any = None):
        super().__init__()
        # projection from inputs to attention logits
        self.q = nn.Linear(d_in, n_heads, bias=False)
        self.q.weight.data.zero_()
        # projection to per-head output logits (or pre-MLP intermediate states)
        self.v = nn.Linear(d_in, n_heads * (hidden_dim or output_dim))
        self.v.weight.data.zero_()
        self.v.bias.data.zero_()

        self.n_heads = n_heads
        self.output_dim = output_dim
        self.use_tanh = use_tanh
        # alibi-like relative (to the beginning/end of the sequence) position bias
        self.position_weight = nn.Parameter(torch.zeros((n_heads,), dtype=torch.float32))
        # MLP after the attention
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.o = nn.Linear(hidden_dim, output_dim)
        # hookpoint to record attention probabilities. use register_forward_hook to record
        self.attn_hook = nn.Identity()
        
        self.config = config

    def forward(self, x, mask, position):
        # x: (batch_size, seq_len, d_in)
        # mask: (batch_size, seq_len)
        # position: (batch_size, seq_len)
        
        # k: (batch_size, seq_len, n_heads)
        # elements that are masked are set to -infinity
        # position is added to the key weighted by the per-head position_weight
        k = self.q(x) - ((1 - mask.float()) * 1e9)[..., None] + position[..., None] * self.position_weight
        # p: (batch_size, seq_len, n_heads)
        # probability of each element after softmax, with masked elements set to 0
        # dim=-2 is the sequence length dimension
        if self.use_tanh:
            p = torch.tanh(k)
        else:
            p = torch.nn.functional.softmax(k, dim=-2)
        # record attention probabilities if necessary
        self.attn_hook(p)
        # v: (batch_size, seq_len, n_heads, output_dim)
        v = self.v(x).unflatten(-1, (self.n_heads, -1))
        # o: (batch_size, output_dim)
        # weight v by the attention probabilities and sum over the sequence length and head dimensions
        o = (p[..., None] * v).sum((-2, -3))
        # if we have an MLP after the attention, apply it
        if self.hidden_dim:
            o = self.o(o.relu())
        return o
