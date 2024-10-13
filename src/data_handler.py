import numpy as np
import torch
from datasets import load_dataset

def prepare_dataset(cfg):
    # Load dataset
    dataset = load_dataset(cfg.dataset.name, split='train')

    # Process data
    processed_data = process_data(dataset, cfg)

    # Split data into train and validation sets
    train_data = {k: v[:int(len(v)*0.8)] for k, v in processed_data.items()}
    val_data = {k: v[int(len(v)*0.8):] for k, v in processed_data.items()}

    return train_data, val_data

def process_data(dataset, cfg):
    processed_data = {
        "image": np.array(dataset["img"]),
        "action": np.concatenate((np.array(dataset["action"]), np.array(dataset["rotation_delta"])), axis=1),
        "goal_image": np.array(dataset["goal_img"]),
        "goal_text": dataset["goal"]
    }

    # Encode text goals
    vocab = create_vocabulary(processed_data["goal_text"])
    processed_data["goal_text"] = encode_text(processed_data["goal_text"], vocab, cfg.model.max_text_length)

    # Normalize actions
    processed_data["action"] = normalize_actions(processed_data["action"])

    # Encode images
    processed_data["image"] = encode_images(processed_data["image"])
    processed_data["goal_image"] = encode_images(processed_data["goal_image"])

    return {k: torch.tensor(v) for k, v in processed_data.items()}

def create_vocabulary(texts):
    chars = sorted(set(char for text in texts for char in text))
    return {char: i for i, char in enumerate(chars)}

def encode_text(texts, vocab, max_length):
    return [
        [vocab.get(char, 0) for char in text[:max_length]] + [0] * (max_length - len(text))
        for text in texts
    ]

def normalize_actions(actions):
    mean = np.mean(actions, axis=0)
    std = np.std(actions, axis=0) + 1e-5
    return (actions - mean) / (std * 1.5)

def encode_images(images):
    return (images / 255.0 * 2.0) - 1.0