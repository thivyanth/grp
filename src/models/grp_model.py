import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Block
from data_loader import get_patches_fast
from models.utils import calc_positional_embeddings

class GRP(nn.Module):
    def __init__(self, dataset, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._dataset = dataset
        self._cfg = cfg
        self.patch_size = (self._cfg.image_shape[0] / self._cfg.n_patches, self._cfg.image_shape[1] / self._cfg.n_patches)
        self.register_buffer('positional_embeddings', calc_positional_embeddings(1 + self._cfg.n_patches ** 2 + self._cfg.block_size + self._cfg.n_patches ** 2, cfg.n_embd), persistent=False)

        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.class_tokens = nn.Parameter(torch.rand(1, cfg.n_embd))

        self.input_d = int(self._cfg.image_shape[2] * self.patch_size[0] * self.patch_size[1])
        self.lin_map = nn.Linear(self.input_d, self._cfg.n_embd, bias=False)
        self.blocks = nn.ModuleList([Block(self._cfg.n_embd, self._cfg.n_head, dropout=self._cfg.dropout) for _ in range(self._cfg.n_blocks)])
        self.mlp = nn.Sequential(
            nn.Linear(self._cfg.n_embd, self._cfg.action_bins),
        )

    def forward(self, images, goals, goal_imgs, targets=None):
        n, c, h, w = images.shape
        B, T = goals.shape
        patches = get_patches_fast(images)
        patches_g = get_patches_fast(goal_imgs)
        goals_e = self.token_embedding_table(goals)
        
        out = self.lin_map(patches)
        out_g = self.lin_map(patches_g)
        
        out = torch.cat((self.class_tokens.expand(n, 1, -1), out, goals_e, out_g), dim=1)
        out = out + self.positional_embeddings.repeat(n, 1, 1)

        mask = torch.ones((1 + c + T + c, ), device=self._cfg.device)
        if targets is None:
            pass
        elif (torch.rand(1)[0] > 0.66):  
            mask[1 + c: 1 + c+ T] = torch.zeros((1,T), device=self._cfg.device)
        elif (torch.rand(1)[0] > 0.33):
            mask[1 + c + T: 1 + c + T + c] = torch.zeros((1,c), device=self._cfg.device)
        
        for block in self.blocks:
            out = block(out, mask)

        out = out[:, 0]
        out = self.mlp(out)
        
        if targets is None:
            loss = None
        else:
            B, C = out.shape
            loss = F.mse_loss(out, targets)
        return (out, loss)