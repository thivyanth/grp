import torch
from omegaconf import OmegaConf
import hydra
from models.grp_model import GRPModel
from data_loader import prepare_data
from trainer import train_model

@hydra.main(config_path="../conf", config_name="grp_config")
def main(cfg):
    print("Configuration:", OmegaConf.to_yaml(cfg))

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create stoi dictionary
    cfg.model.stoi = {ch: i for i, ch in enumerate(cfg.vocab.chars)}

    # Prepare data
    train_loader, val_loader = prepare_data(cfg)

    # Initialize model
    model = GRPModel(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Train model
    train_model(model, train_loader, val_loader, cfg.training, device)

if __name__ == "__main__":
    main()
