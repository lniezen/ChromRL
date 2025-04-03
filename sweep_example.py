import os
import wandb
import numpy as np
import torch as th
import torch.nn as nn
import gc

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList

from CustomEvalCallback import CustomEvalCallback
from CustomLoggingCallback import TensorboardCallback
from CustomLRSchedule import *
from wandb.integration.sb3 import WandbCallback

from env import Chromatography


# 1: Define objective/training function
def main():
    standard_config = model_config | env_config | extra_config

    run = wandb.init(
        config=model_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Create directories for logging, checkpoints, and evaluation
    models_dir = f"models/{run.id}/"
    logs_dir = f"logs/{run.id}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    filename = "sample_set_10p.npy"  # Ensure this is the correct filename
    solvable_mixtures = np.load(filename, allow_pickle=True).item()

    # Make the training environment
    env = make_vec_env(Chromatography,
                       n_envs=standard_config["number_of_parallel_envs"],
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs={'time': standard_config["maximum_experiment_duration"],
                                   'time_target': standard_config["time_target"],
                                   'chromatogram_size': standard_config["chromatogram_datapoints"],
                                   'num_experiments': wandb.config["number_of_experiments"],
                                   'num_compounds': standard_config["number_of_compounds"],
                                   'num_actions': standard_config["number_of_segments"] + 1,
                                   'seed': 0,
                                   "solvable_mixtures": solvable_mixtures
                                   })
    # Make a separate evaluation environment
    eval_env = make_vec_env(Chromatography,
                            n_envs=standard_config["number_of_parallel_envs"],
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={'time': standard_config["maximum_experiment_duration"],
                                        'time_target': standard_config["time_target"],
                                        'chromatogram_size': standard_config["chromatogram_datapoints"],
                                        'num_experiments': 20,
                                        'num_compounds': standard_config["number_of_compounds"],
                                        'num_actions': standard_config["number_of_segments"] + 1,
                                        'seed': None,
                                        "solvable_mixtures": solvable_mixtures
                                        })

    # Stack consecutive observations together, given a particular number of frames (n_stack)
    env = VecFrameStack(env, n_stack=wandb.config["number_of_stacked_observations"], channels_order="last")
    eval_env = VecFrameStack(eval_env, n_stack=wandb.config["number_of_stacked_observations"], channels_order="last")

    # Specify (MLP) network architecture, n_layers and nodes for actor (pi) and critic (vf) networks
    pi_layer_size = wandb.config["pi_layer_size"]
    pi_num_layers = wandb.config["pi_num_layers"]
    vf_layer_size = wandb.config["vf_layer_size"]
    vf_num_layers = wandb.config["vf_num_layers"]

    net_arch = {
        "pi": [pi_layer_size] * pi_num_layers,  # Actor network architecture
        "vf": [vf_layer_size] * vf_num_layers  # Critic network architecture
    }

    policy_kwargs = dict(
        # features_extractor_class=CustomCombinedExtractor, # For implementing a custom feature extractor, e.g. a CNN
        activation_fn=nn.LeakyReLU,  # Activation function
        optimizer_class=th.optim.Adam,  # Optimizer class
        optimizer_kwargs=dict(eps=2.5e-5),  # Learning rate for the optimizer
        net_arch=net_arch, # Network architecture
        log_std_init=-2, # Initial action std
    )

    # Copy sweep config and remove any non-model-specific parameters before passing to the model
    model_parameters = wandb.config.as_dict().copy()
    '''
    model_parameters.pop("pi_layer_size")
    model_parameters.pop("pi_num_layers")
    model_parameters.pop("vf_layer_size")
    model_parameters.pop("vf_num_layers")
    model_parameters.pop("number_of_experiments")
    model_parameters.pop("initial_learning_rate")
    model_parameters.pop("number_of_stacked_observations")
    '''
    keys_to_remove = {
        "pi_layer_size", "pi_num_layers", "vf_layer_size", "vf_num_layers",
        "number_of_experiments", "initial_learning_rate", "number_of_stacked_observations"
    }

    for key in keys_to_remove:
        model_parameters.pop(key, None)

    # Initialize the model
    model = PPO(env=env,
                learning_rate=custom_lr_schedule(wandb.config["initial_learning_rate"], 1e-10,
                                                 total_steps=extra_config["total_training_steps"],
                                                 decay_steps=25_000_000),
                normalize_advantage=True,
                use_sde=False,
                sde_sample_freq=-1,
                stats_window_size=100,
                tensorboard_log=logs_dir,
                _init_setup_model=True,
                policy_kwargs=policy_kwargs,
                **model_parameters)


    # Create callback for logging, checkpoints and evaluation and anything else you would want
    logging_callback = TensorboardCallback(verbose=0)
    eval_callback = CustomEvalCallback(eval_env,
                                 eval_freq=4_096,
                                 n_eval_episodes=500,
                                 best_model_save_path=models_dir,
                                 verbose=1)
    callbacks = CallbackList([
        logging_callback,
        eval_callback,
        WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2)
    ])

    model.learn(total_timesteps=extra_config["total_training_steps"],
                reset_num_timesteps=True,
                callback=callbacks)

    # Evaluate and log the performance of the best performing model in this run
    best_model_path = models_dir + "best_model"
    best_model = PPO.load(best_model_path)
    mean_reward, _ = evaluate_policy(best_model, eval_env, n_eval_episodes=500)
    run.log({"mean_reward": mean_reward})

    del model
    del best_model
    th.cuda.empty_cache()
    gc.collect()


# Specify default config and standard hyperparams
model_config = {
    # Policy and generic training settings
    "policy": "MultiInputPolicy",  # This is determined by what observation space you use.
    "device": "cuda",              # Performs network updates on GPU, if available
    "seed": 42,
    "verbose": 0,                  # Whether to write training progress or not

    # Hyperparameters, model specific
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "clip_range": 0.2,
    "clip_range_vf": 0.2,
    "max_grad_norm": 0.5,
    "target_kl": 0.01
}

env_config = {
    # Environment settings
    "maximum_experiment_duration": 20.0,
    "time_target": 20.0,
    "chromatogram_datapoints": 8192,
    "number_of_compounds": (10, 10),
    "number_of_segments": 1,
    "render_mode": 'rgb_array',
    "env_seed": None
}

extra_config = {
    # Additional settings
    "run_id": "Example_run_id",
    "tb_log_name": f"RLChrom",
    "iterations": 100,
    "total_training_steps": 5_000_000,  # Total training interactions
    "number_of_parallel_envs": 16,      # Training in n parallel envs. running on separate threads
}

# Define the sweep search space, overwrites default config. when sweeping
# Any parameter can be sweeped over (e.g. num. stacked observations) by removing them before passing them to the model.
sweep_configuration = {
    "name":
        "Example_sweep",
    "method":
        "grid", # Specify random, grid or bayes here for the type of search to use
    "metric":
        {"goal": "maximize", "name": "mean_reward"},
    "parameters":
        {
            "number_of_stacked_observations":
                {"values": [10]},
            "number_of_experiments":
                {"values": [200]},
            "initial_learning_rate":
                {"values": [1e-5]},
            "n_steps":
                {"values": [64]},
            "batch_size":
                {"values": [128]},
            "n_epochs":
                {"values": [5]},
            "gamma":
                {"values": [0.95]},
            "gae_lambda":
                {"values": [0.90]},
            "clip_range":
                {"values": [0.5]},
            "clip_range_vf":
                {"values": [1.0]},
            "ent_coef":
                {"values": [0.0]},
            "vf_coef":
                {"values": [1.0]},
            "max_grad_norm":
                {"values": [0.5]},

            "pi_num_layers": {
                "values": [2]},     # num. of actor  network layers
            "pi_layer_size": {
                "values": [256]},   # num. of actor (pi) layer nodes (fixed size throughout)
            "vf_num_layers": {
                "values": [2]},     # num. of critic  network layers
            "vf_layer_size": {
                "values": [128]},   # num. of critic (vf) layer nodes (fixed size throughout)
        }
}

# Run the tuning sweep and store run graphs in wandb project
if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="PPO_Chromatography_Project")
    wandb.agent(sweep_id=sweep_id, function=main)
