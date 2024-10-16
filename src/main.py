import torch
from omegaconf import OmegaConf
import hydra
from models.grp_model import GRPModel
from data_loader import prepare_data
from trainer import train_model
import wandb

@hydra.main(config_path="../conf", config_name="grp_config")
def main(cfg):
    
    # Control Panel
    cfg.debug = True
    cfg.training.batch_size = 4

    print("Configuration:", OmegaConf.to_yaml(cfg))
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, action_mean, action_std = prepare_data(cfg)

    # Initialize model
    model = GRPModel(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Initialize wandb if not in debug mode
    if not cfg.debug:
        wandb.init(project="grp-project", config=OmegaConf.to_container(cfg, resolve=True))

    # Train model
    train_model(model, train_loader, val_loader, cfg, device, action_mean, action_std)

    # Close wandb
    if not cfg.debug:
        wandb.finish()

if __name__ == "__main__":
    main()
