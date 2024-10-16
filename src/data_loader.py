import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms

class RoboticsDataset(Dataset):
    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg

    def __len__(self):
        return len(self.data['img'])

    def __getitem__(self, idx):
        return (self.data['img'][idx], 
                self.data['goal'][idx], 
                self.data['goal_img'][idx], 
                self.data['action'][idx])

def prepare_data(cfg):
    dataset = load_dataset(cfg.data.dataset, split='train')
    print('Features:', dataset.features)

    dataset_tmp = {
        "img": np.array(dataset["img"]),
        "action": np.concatenate((np.array(dataset["action"]), 
                                  np.array(dataset["rotation_delta"])), axis=1),
        "goal_img": np.array(dataset["goal_img"]),
        "goal": dataset["goal"]
    }

    shortest_text_len = min([len(txt) for txt in dataset["goal"]])
    cfg.data.block_size = shortest_text_len

    # Create character to integer mapping
    chars = sorted(list(set([item for row in dataset_tmp["goal"] for item in row])))
    cfg.data.vocab_size = len(chars)
    cfg.stoi = {ch: i for i, ch in enumerate(chars)}
    cfg.itos = {i: ch for i, ch in enumerate(chars)}

    encode_txt = lambda s: [cfg.stoi[c] for c in s[:cfg.data.block_size]]
    
    # Calculate action mean and std
    action_mean = np.mean(dataset_tmp['action'], axis=0)
    action_std = np.std(dataset_tmp['action'], axis=0) * 1.5 + 0.001

    resize = transforms.Resize((cfg.data.image_shape[0], cfg.data.image_shape[1]))
    encode_state = lambda af: resize(transforms.ToPILImage()(((af/255.0)*2.0)-1.0))
    
    encode_action = lambda af: (((af - action_mean) / (action_std))).astype(np.float32)

    dataset_processed = {
        "img": torch.stack([transforms.ToTensor()(encode_state(img)) for img in dataset_tmp["img"]]),
        "action": torch.tensor(encode_action(dataset_tmp["action"]), dtype=torch.float),
        "goal_img": torch.stack([transforms.ToTensor()(encode_state(img)) for img in dataset_tmp["goal_img"]]),
        "goal": torch.tensor([encode_txt(goal[:cfg.data.block_size]) for goal in dataset_tmp["goal"]])
    }

    # Create datasets
    train_dataset = RoboticsDataset(dataset_processed, cfg)
    val_dataset = RoboticsDataset(dataset_processed, cfg)  # In practice, use a separate validation set

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    return train_loader, val_loader, action_mean, action_std
