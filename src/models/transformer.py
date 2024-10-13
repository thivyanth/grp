import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
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
