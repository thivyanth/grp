import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .attention import FeedFoward

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.feed_forward = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_embd, n_head, n_layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
