# Reinforcement learning for automated method development in chromatography using SB3 PPO
## Requirements: 
* Python >= 3.10
* Stable-baselines3 = 2.5.0
* Pytorch = 2.6.0 + Cuda 12.4
* Wandb = 0.19.4 
* All required Python packages can be found and installed via the environment.yml
* Install the relevant Pytorch version via: https://pytorch.org/get-started/locally/?ajs_aid=f5f8111c-aeea-415a-b925-871de30f72bc

## Description + Training:
This is further work aimed at training reinforcement learning agents with the purpose of automating chromatographic method development.  
The algorithms used are reliable stable-baselines 3 (PyTorch) implementations of reinforcement learning algorithms + weight and biases for monitoring and tuning.  

To train an example PPO agent navigate to and run the sweep_example code, which trains a PPO agent capable of selecting 20 min. gradient programs for simple samples containing 10 components.  

For more information regarding implementation details, see this publication:  
https://doi.org/10.1016/j.chroma.2025.465845  

The original environment (and the work which this is heavily based on) can be found at:  
https://github.com/akensert/reinforceable  
