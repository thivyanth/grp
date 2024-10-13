# Generalist Robotics Policy (GRP) Project

This project implements a Generalist Robotics Policy (GRP) using a Vision Transformer (ViT) architecture. The model is designed to process multiple input types, including images, text goals, and goal images, to generate continuous action outputs for robotic control.

## Current Status

**Note: This project is currently in the debugging phase.**

I'm working on resolving issues, understanding and improving the model.

## Getting Started

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create and activate a conda environment:
   ```
   conda create -n grp python=3.10
   conda activate grp
   ```
4. Install PyTorch with CUDA support:
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
5. Install additional dependencies:
   ```
   pip install torch==2.4.0
   pip install hydra-submitit-launcher --upgrade
   pip install milatools==0.1.14 decorator==4.4.2 moviepy==1.0.3
   ```
6. Run the main script:
   ```
   python src/main.py
   ```

## TODO

- [ ] Complete debugging of the main training loop
- [ ] Implement evaluation in a simulation environment
- [ ] Implement logging and visualization tools
