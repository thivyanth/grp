import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import cv2
import time
import os
from datasets import load_dataset
from data_loader import get_batch_grp

from models.grp_model import GRP
from models.utils import estimate_loss

@hydra.main(config_path="../conf", config_name="grp_config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    print("cfg:", OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    cfg.device = device

    dataset = load_dataset(cfg.dataset, split='train')
    print('Features:', dataset.features)

    dataset_tmp = prepare_dataset(dataset, cfg)
    
    if not cfg.testing:
        import wandb
        wandb.init(project="grp", config=OmegaConf.to_container(cfg))
        wandb.run.log_code(".")
    
    model = GRP(dataset_tmp, cfg).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)

    train_model(model, optimizer, scheduler, dataset_tmp, cfg)

    if not cfg.testing:
        wandb.finish()

def prepare_dataset(dataset, cfg):
    dataset_tmp = {
        "img": np.array(dataset["img"]),
        "action": np.concatenate((np.array(dataset["action"]), np.array(dataset["rotation_delta"])), axis=1),
        "goal_img": np.array(dataset["goal_img"]),
        "goal": dataset["goal"]
    }
    
    cfg.block_size = min([len(txt) for txt in dataset["goal"]])
    chars = sorted(list(set([item for row in dataset_tmp["goal"] for item in row])))
    cfg.vocab_size = len(chars)
    
    stoi = {ch:i for i,ch in enumerate(chars)}
    encode_txt = lambda s: [stoi[c] for c in s]
    
    a_std, a_mean = (dataset_tmp["action"].std(axis=0) + 0.001) * 1.5, dataset_tmp["action"].mean(axis=0)
    cfg.action_bins = len(a_mean)
    encode_action = lambda af: (((af - a_mean)/(a_std))).astype(np.float32)
    encode_state = lambda af: ((af/(255.0)*2.0)-1.0).astype(np.float32)
    
    dataset_tmp = {
        "img": torch.tensor(encode_state(dataset_tmp["img"])).to(cfg.device),
        "action": torch.tensor(encode_action(dataset_tmp["action"]), dtype=torch.float).to(cfg.device),            
        "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"])).to(cfg.device),
        "goal": torch.tensor([encode_txt(goal[:cfg.block_size]) for goal in dataset_tmp["goal"]]).to(cfg.device)
    }
    
    return {"train": dataset_tmp, "test": dataset_tmp}

def train_model(model, optimizer, scheduler, dataset, cfg):
    start_time = time.time()
    
    for iter in range(cfg.max_iters):
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, cfg)
            log_progress(iter, losses, start_time, cfg)
        
        xb, xg, xgi, yb = get_batch_grp('train', dataset, cfg.batch_size)
        logits, loss = model(xb, xg, xgi, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    save_checkpoint(model, iter, cfg)

def log_progress(iter, losses, start_time, cfg):
    elapsed_time = time.time() - start_time
    eta = (elapsed_time / (iter + 1)) * (cfg.max_iters - iter - 1)
    
    print(f"step {iter}/{cfg.max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f}s, ETA: {eta:.2f}s")
    
    if not cfg.testing:
        import wandb
        wandb.log({
            "train loss": losses['train'],
            "val loss": losses['val'],
            "iteration": iter,
            "progress": iter / cfg.max_iters,
            "elapsed_time": elapsed_time,
            "eta": eta
        })

def save_checkpoint(model, iter, cfg):
    checkpoint_path = os.path.join("outputs", f'checkpoint_iter_{iter}.pt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model, checkpoint_path)
    print(f"Saved checkpoint at iteration {iter} to {checkpoint_path}")

if __name__ == "__main__":
    main()
