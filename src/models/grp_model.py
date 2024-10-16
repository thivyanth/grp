import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .utils import calc_positional_embeddings  # Adjust the import path as needed
import torch.nn.functional as F

class GRPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.token_embedding = nn.Embedding(cfg.data.vocab_size, cfg.model.embed_dim)
        self.init_positional_embedding()

        # Image processing
        self.image_patch_embedder = ImagePatchEmbedder(cfg)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg.model.embed_dim, cfg.model.num_heads, cfg.model.dropout)
            for _ in range(cfg.model.num_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(cfg.model.embed_dim, cfg.action_dim)

    def init_positional_embedding(self):
        max_length = self.cfg.data.block_size + (self.cfg.data.image_shape[0] // self.cfg.model.patch_size) ** 2 * 2
        d = self.cfg.model.embed_dim
        
        positional_embeddings = calc_positional_embeddings(max_length, d)
        
        self.register_buffer('positional_embedding', positional_embeddings.unsqueeze(0))

    def forward(self, images, goals, goal_images, targets=None):
        # Process inputs
        img_tokens = self.image_patch_embedder(images)
        goal_img_tokens = self.image_patch_embedder(goal_images)
        goal_tokens = self.token_embedding(goals)

        # Combine tokens
        tokens = torch.cat([img_tokens, goal_tokens, goal_img_tokens], dim=1)
        tokens = tokens + self.positional_embedding[:, :tokens.size(1), :]

        # Create mask
        mask = self.create_mask(img_tokens, goal_tokens, goal_img_tokens)

        # Apply transformer blocks with mask
        for block in self.transformer_blocks:
            tokens = block(tokens, mask)

        # Get output from the first token
        output = self.output_layer(tokens[:, 0])

        if targets is not None:
            loss = F.mse_loss(output, targets)
            return output, loss
        return output

    def create_mask(self, img_tokens, goal_tokens, goal_img_tokens):
        batch_size = img_tokens.size(0)
        img_len = img_tokens.size(1)
        goal_len = goal_tokens.size(1)
        goal_img_len = goal_img_tokens.size(1)
        total_len = img_len + goal_len + goal_img_len

        mask = torch.ones((batch_size, total_len), device=self.cfg.device)

        if self.training:
            # Randomly mask goal tokens or goal image tokens
            rand = torch.rand(1).item()
            if rand > 0.66:
                mask[:, img_len:img_len+goal_len] = 0  # Mask goal string
            elif rand > 0.33:
                mask[:, img_len+goal_len:] = 0  # Mask goal image

        return mask.unsqueeze(1)  # Add dimension for attention heads

class ImagePatchEmbedder(nn.Module):
    """
    ImagePatchEmbedder is responsible for processing the image data into tokens.
    It divides the image into patches and projects them into the embedding space.
    """
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.model.patch_size
        self.image_shape = cfg.data.image_shape
        self.num_patches = (self.image_shape[0] // self.patch_size) ** 2
        self.projection = nn.Linear(3 * self.patch_size ** 2, cfg.model.embed_dim)

    def forward(self, x):
        # x: (batch_size, 3, H, W)
        batch_size = x.shape[0]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, self.num_patches, -1)
        return self.projection(patches)
