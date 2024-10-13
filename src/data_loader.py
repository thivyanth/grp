import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

class RoboticsDataset(Dataset):
    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg
        self.encode_txt = lambda s: [self.cfg.stoi[c] for c in s[:self.cfg.block_size]]
        self.encode_state = lambda af: ((af/(255.0)*2.0)-1.0).astype(np.float32)
        self.encode_action = lambda af: (((af - self.cfg.action_mean) / (self.cfg.action_std))).astype(np.float32)

    def __len__(self):
        return len(self.data['img'])

    def __getitem__(self, idx):
        img = torch.tensor(self.encode_state(self.data['img'][idx]))
        action = torch.tensor(self.encode_action(self.data['action'][idx]))
        goal_img = torch.tensor(self.encode_state(self.data['goal_img'][idx]))
        goal = torch.tensor(self.encode_txt(self.data['goal'][idx]))
        
        return img, action, goal_img, goal

def prepare_data(cfg):
    # Load your data here (e.g., from a file or API)
    # For now, we'll use dummy data
    dataset = {
        'img': np.random.randint(0, 256, (1000, 64, 64, 3), dtype=np.uint8),
        'action': np.random.randn(1000, cfg.action_dim),
        'goal': [''.join(np.random.choice(list(cfg.stoi.keys()), 20)) for _ in range(1000)]
    }

    # Calculate action mean and std
    cfg.action_mean = np.mean(dataset['action'], axis=0)
    cfg.action_std = np.std(dataset['action'], axis=0) * 1.5 + 0.001

    # Create datasets
    train_dataset = RoboticsDataset(dataset, cfg)
    val_dataset = RoboticsDataset(dataset, cfg)  # In practice, use a separate validation set

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader
