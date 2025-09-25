import os
import wandb
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList

from utils.CustomEvalCallback import CustomEvalCallback
from utils.CustomLoggingCallback import TensorboardCallback
from utils.CustomLRSchedule import *
from wandb.integration.sb3 import WandbCallback

# from envs.env_chromatogram_obs import Chromatography
from envs.env_peak_obs import Chromatography
# from envs.env_mixed_obs import Chromatography

def main():
    default_config = model_config | env_config | extra_config

    # Initialize W&B with defaults, sweeps overwrite as needed
    run = wandb.init(
        config=default_config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )
    cfg = dict(wandb.config)

    # Create directories for logging, checkpoints, and evaluation
    models_dir = f"models/{run.id}/"
    logs_dir = f"logs/{run.id}/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load solvable mixtures
    filename = "sample_sets/example_samples_15p.npy"
    solvable_mixtures = np.load(filename, allow_pickle=True).item()

    # Training environment
    env = make_vec_env(
        Chromatography,
        n_envs=cfg["number_of_parallel_envs"],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "time": cfg["maximum_experiment_duration"],
            "time_target": cfg["time_target"],
            "chromatogram_size": cfg["chromatogram_datapoints"],
            "num_experiments": cfg["number_of_experiments"],
            "num_compounds": cfg["number_of_compounds"],
            "num_actions": cfg["number_of_segments"] + 1,
            "seed": 0,
            "solvable_mixtures": solvable_mixtures,
        },
    )

    # Evaluation environment
    eval_env = make_vec_env(
        Chromatography,
        n_envs=cfg["number_of_parallel_envs"],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "time": cfg["maximum_experiment_duration"],
            "time_target": cfg["time_target"],
            "chromatogram_size": cfg["chromatogram_datapoints"],
            "num_experiments": 20,
            "num_compounds": cfg["number_of_compounds"],
            "num_actions": cfg["number_of_segments"] + 1,
            "seed": None,
            "solvable_mixtures": solvable_mixtures,
        },
    )

    # Stack observations (happens according to a FIFO principle)
    env = VecFrameStack(env, n_stack=cfg["number_of_stacked_observations"], channels_order="last")
    eval_env = VecFrameStack(eval_env, n_stack=cfg["number_of_stacked_observations"], channels_order="last")

    # Build net architecture
    net_arch = {
        "pi": [cfg["pi_layer_size"]] * cfg["pi_num_layers"],
        "vf": [cfg["vf_layer_size"]] * cfg["vf_num_layers"],
    }

    policy_kwargs = dict(
        # features_extractor_class=CustomCombinedExtractor, # For implementing a custom feature extractor, e.g. a CNN
        activation_fn=nn.LeakyReLU,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=dict(eps=2.5e-5),
        net_arch=net_arch,
        log_std_init=-2,
    )

    # Take just the model-related keys from the config to prevent errors
    model_parameters = {k: cfg[k] for k in model_config.keys() if k in cfg}

    # Initialize PPO
    model = PPO(
        env=env,
        learning_rate=custom_lr_schedule(
            cfg["initial_learning_rate"],
            1e-7,
            total_steps=cfg["total_training_steps"],
            decay_steps=25_000_000,
        ),
        normalize_advantage=True,
        use_sde=False,
        sde_sample_freq=-1,
        stats_window_size=100,
        tensorboard_log=logs_dir,
        _init_setup_model=True,
        policy_kwargs=policy_kwargs,
        **model_parameters,
    )

    # Callbacks
    logging_callback = TensorboardCallback(verbose=0)
    eval_callback = CustomEvalCallback( # CustomEval to include additional logs other than just reward and time steps
        eval_env,
        eval_freq=4_096,
        n_eval_episodes=500,
        best_model_save_path=models_dir,
        verbose=1,
    )
    callbacks = CallbackList([
        logging_callback,
        eval_callback,
        WandbCallback(model_save_path=f"models/{run.id}", verbose=2),
    ])

    # Train the model
    model.learn(
        total_timesteps=cfg["total_training_steps"],
        reset_num_timesteps=True,
        callback=callbacks,
    )

    # Do a final evaluation of the best found model
    best_model_path = os.path.join(models_dir, "best_model")
    best_model = PPO.load(best_model_path)
    mean_reward, _ = evaluate_policy(best_model, eval_env, n_eval_episodes=1000)
    run.log({"mean_reward": mean_reward})


# Specify the default config (model, env. etc.) and standard hyperparams
model_config = {
    # Policy and generic training settings
    "policy": "MultiInputPolicy",  # This is determined by what observation space you use.
    "device": "cuda",              # Performs network updates on GPU, if available
    "seed": 42,
    "verbose": 0,                  # Whether to write training progress or not

    # Hyperparameters, model specific. Note: if no defaults are specified here then SB3's defaults are used
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
    "number_of_compounds": (15, 15),
    "number_of_segments": 1,
    "render_mode": 'rgb_array',
    "env_seed": None
}

extra_config = {
    # Additional settings
    "run_id": "Example_run_id",
    "tb_log_name": f"RLChrom",
    "total_training_steps": 5_000_000,  # Total training interactions
    "number_of_parallel_envs": 16,      # Keep below num. of cores available.
}

# Define the sweep search space, overwrites the default config. when sweeping
# Any parameter can be sweeped over (e.g. num. stacked observations or env. parameters)
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