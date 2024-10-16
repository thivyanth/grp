import torch
import torch.nn as nn
from .transformer import TransformerBlock

class GRPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

        # Embeddings
        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.embed_dim)
        self.positional_embedding = self.create_positional_embedding()

        # Image processing
        self.image_processor = ImageProcessor(cfg)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.cfg.embed_dim, self.cfg.num_heads, self.cfg.dropout)
            for _ in range(self.cfg.num_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(self.cfg.embed_dim, self.cfg.action_dim)

    def forward(self, images, goals, goal_images):
        # Process inputs
        img_tokens = self.image_processor(images)
        goal_img_tokens = self.image_processor(goal_images)
        goal_tokens = self.token_embedding(goals)

        # Combine tokens
        tokens = torch.cat([img_tokens, goal_tokens, goal_img_tokens], dim=1)
        tokens = tokens + self.positional_embedding[:tokens.size(1)]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Get output from the first token
        output = self.output_layer(tokens[:, 0])

        return output

    def create_positional_embedding(self):
        # Implement positional embedding creation here
        # For now, we'll use a simple learnable embedding
        max_length = self.cfg.block_size + self.cfg.n_patches**2 * 2  # Adjust as needed
        return nn.Parameter(torch.randn(1, max_length, self.cfg.embed_dim))

class ImageProcessor(nn.Module):
    """
    ImageProcessor is responsible for processing the image data into tokens.
    It divides the image into patches and projects them into the embedding space.
    """
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.model.patch_size
        self.num_patches = (cfg.data.image_size // self.patch_size) ** 2
        self.projection = nn.Linear(3 * self.patch_size ** 2, cfg.model.embed_dim)
    def forward(self, x):
        # x: (batch_size, 3, H, W)
        batch_size = x.shape[0]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, self.num_patches, -1)
        return self.projection(patches)
