import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import cv2
from models.grp_model import GRP
import simpler_env
import os
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# export MODEL_PATH="mini-grp/outputs/2024-10-16/15-53-01/outputs/checkpoint_iter_499.pt"

os.environ['DISPLAY'] = ':0'

@hydra.main(config_path="../conf", config_name="grp_config")
def simulate(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    print("cfg:", OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    cfg.device = device

    # Load the saved model
    model_path = os.environ.get('MODEL_PATH')
    print("model_path:", model_path)
    model = torch.load(model_path)
    model.eval()
    model.to(device)

    # Load environment
    task_name = "widowx_carrot_on_plate"
    task_name = "google_robot_pick_coke_can"
    env = simpler_env.make(task_name)

    # Load encoding functions
    encode_state, encode_txt, decode_action = load_encoding_functions(cfg)

    for episode in [0]: # range(cfg.num_episodes):
        obs, reset_info = env.reset()
        instruction = env.unwrapped.get_language_instruction()
        print("Reset info", reset_info)
        print("Instruction", instruction)
        frames, rewards = [], []
        done, truncated = False, False

        while not (done or truncated):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            
            # Prepare inputs for the model
            resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))
            image_tensor = torch.tensor(np.array([encode_state(resize_state(image))])).to(device)
            instruction_tensor = torch.tensor(np.array([encode_txt(instruction)[:cfg.block_size]])).to(device)

            # Get action from model
            with torch.no_grad():
                action, _ = model.forward(image_tensor, instruction_tensor)

            # Decode action
            action = np.concatenate((decode_action(action.cpu().detach().numpy()[0]), [0]), axis=-1)

            # Take action in environment
            obs, reward, done, truncated, info = env.step(action)
            frames.append(image)
            rewards.append(reward)

        # Log episode results
        episode_stats = info.get('episode_stats', {})
        print("Episode stats", episode_stats)
        print(f"avg reward {np.mean(rewards):.8f}")

        # Save video of the episode
        save_video(frames, episode)

def load_encoding_functions(cfg):
    # Load necessary encoding functions (you may need to adjust this based on your actual implementation)
    encode_state = lambda af: ((af/(255.0)*2.0)-1.0).astype(np.float32)
    resize_state = lambda sf: cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))

    # You'll need to implement these functions based on your model's requirements
    encode_txt = lambda s: [0]  # Placeholder, replace with actual implementation
    decode_action = lambda binN: binN  # Placeholder, replace with actual implementation

    return encode_state, encode_txt, decode_action

def save_video(frames, episode):
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(list(frames), fps=20)
    clip.write_videofile(f"./data/sim-env-{episode}.mp4", fps=20)

if __name__ == "__main__":
    simulate()
