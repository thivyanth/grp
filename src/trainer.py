import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_model(model, train_loader, val_loader, cfg, device):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(cfg.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{cfg.num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def create_dataloader(data, batch_size):
    dataset = TensorDataset(data['image'], data['goal_text'], data['goal_image'], data['action'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, actions, goal_images, goals in tqdm(dataloader, desc="Training"):
        images, actions, goal_images, goals = (images.to(device), actions.to(device),
                                               goal_images.to(device), goals.to(device))

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
        for images, actions, goal_images, goals in tqdm(dataloader, desc="Evaluating"):
            images, actions, goal_images, goals = (images.to(device), actions.to(device),
                                                   goal_images.to(device), goals.to(device))

            outputs = model(images, goals, goal_images)
            loss = criterion(outputs, actions)
            total_loss += loss.item()

    return total_loss / len(dataloader)
