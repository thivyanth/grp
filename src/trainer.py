import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import os

def train_model(model, train_loader, val_loader, cfg, device, action_mean, action_std):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(cfg.training.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg.debug)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{cfg.training.num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if not cfg.debug:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if (epoch + 1) % cfg.training.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg)

def train_epoch(model, dataloader, optimizer, criterion, device, debug):
    model.train()
    total_loss = 0

    for images, goals, goal_images, actions in tqdm(dataloader, desc="Training", disable=debug):
        images, goals, goal_images, actions = (images.to(device), goals.to(device),
                                               goal_images.to(device), actions.to(device))

        optimizer.zero_grad()
        outputs = model(images, goals, goal_images)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, goals, goal_images, actions in dataloader:
            images, goals, goal_images, actions = (images.to(device), goals.to(device),
                                                   goal_images.to(device), actions.to(device))

            outputs = model(images, goals, goal_images)
            loss = criterion(outputs, actions)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    checkpoint_path = os.path.join(cfg.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
