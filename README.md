# Generalist Robotics Policy (GRP) Project

This project is my implementation of a Generalist Robotics Policy (GRP) using a Vision Transformer (ViT) architecture. Inspired by existing approaches but built from the ground up, this model processes multiple input types such as images, text goals, and goal images to generate continuous action outputs for robotic control. While drawing inspiration from established concepts, I've recreated this implementation to deepen my understanding of GRP architectures. This is basically a mini and very basic version of [octo](https://github.com/octo-models/octo).

## Getting Started

1. Clone the repository
2. Create and activate a conda environment:
   ```
   conda create -n grp python=3.10
   conda activate grp
   ```
3. Install PyTorch with CUDA support:
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
4. Install additional dependencies:
   ```
   pip install torch==2.4.0
   pip install hydra-submitit-launcher --upgrade
   pip install decorator==4.4.2 moviepy==1.0.3
   ```
5. Install the required project dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Run the main script:
   ```
   python src/main.py
   ```

## TODO

- [x] Complete debugging of the main training loop
- [x] Create evaluation in a simulation environment
- [ ] Debug evaluation in a simulation environment
- [ ] Visualization tools

## Extra Challenges

- [ ] Incorporate Diffusion Models
- [ ] Use Text Tokenization like BPE

## Acknowledgements
- [mini-GRP](https://github.com/milarobotlearningcourse/mini-grp)
- [octo](https://github.com/octo-models/octo)
- [nano-GPT](https://github.com/karpathy/nanoGPT)