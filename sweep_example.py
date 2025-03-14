import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import (CallbackList,
                                                EvalCallback)
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import torch as th
import torch.nn as nn
import gc

from env import Chromatography
from typing import Callable
import os


def custom_lr_schedule(initial_value: float, final_value: float, total_steps: int, decay_steps: int) -> Callable[
    [float], float]:
    """
    Linear learning rate schedule.

    :param total_steps: Total training steps
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :param decay_steps: Number of steps in which to decay from initial to final lr.

    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        decay_rate = total_steps / decay_steps
        current_progress = (1 - progress_remaining)
        current_learning_rate = initial_value - (initial_value - final_value) * current_progress * decay_rate
        if current_learning_rate < final_value:
            current_learning_rate = final_value

        return current_learning_rate

    return func


class TensorboardCallback(BaseCallback):
    """
    Logging function to write additional environment data to Tensorboard/WandB
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_info_buffer = []

    def _on_step(self) -> bool:
        # Get the `dones` flag and `infos`
        done_flags = self.locals['dones']
        infos = self.locals['infos']

        # Check if the episode has ended
        for i in range(len(done_flags)):
            if done_flags[i]:
                # Extract and store relevant information from `infos` at episode end
                self.episode_info_buffer.append(infos[i]['Solved'])

        return True

    def _on_rollout_end(self):
        # Compute and log the mean of collected infos if there is data
        if len(self.episode_info_buffer) > 0:
            mean_solved = np.mean(self.episode_info_buffer)
            self.logger.record("mean_solved", mean_solved)

            # Clear the buffer after logging
            self.episode_info_buffer.clear()

        # Dump logs to tensorboard at the end of the rollout
        self.logger.dump(self.num_timesteps)


# 1: Define objective/training function
def main():
    standard_config = model_config | env_config | extra_config
    run = wandb.init(
        config=model_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    # Create directories for logging, checkpoints, and evaluation etc.
    models_dir = f"models/{run.id}/"
    logs_dir = f"logs/{run.id}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    filename = "results_10p.npy"
    solvable_mixtures = np.load(filename, allow_pickle='TRUE').item()

    # Make the training environment (known, solvable samples)
    env = make_vec_env(Chromatography,
                       n_envs=standard_config["number_of_parallel_envs"],
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs={'time': standard_config["maximum_experiment_duration"],
                                   'time_target': standard_config["maximum_experiment_duration"],
                                   'chromatogram_size': standard_config["chromatogram_datapoints"],
                                   'num_experiments': wandb.config["number_of_experiments"],
                                   'num_compounds': standard_config["number_of_compounds"],
                                   'num_actions': standard_config["number_of_segments"] + 1,
                                   'seed': 0,
                                   "solvable_mixtures": solvable_mixtures
                                   })
    # Make the evaluation environment (random solvable/unsolvable samples)
    eval_env = make_vec_env(Chromatography,
                            n_envs=standard_config["number_of_parallel_envs"],
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={'time': standard_config["maximum_experiment_duration"],
                                        'time_target': standard_config["maximum_experiment_duration"],
                                        'chromatogram_size': standard_config["chromatogram_datapoints"],
                                        'num_experiments': 20,
                                        'num_compounds': standard_config["number_of_compounds"],
                                        'num_actions': standard_config["number_of_segments"] + 1,
                                        'seed': None,
                                        "solvable_mixtures": solvable_mixtures
                                        })

    # Stack consecutive observations together using a framestack wrapper
    env = VecFrameStack(env, n_stack=wandb.config["number_of_stacked_observations"], channels_order="last")
    eval_env = VecFrameStack(eval_env, n_stack=wandb.config["number_of_stacked_observations"], channels_order="last")

    policy_kwargs = dict(
        # features_extractor_class=CustomCombinedExtractor, #If you want to use a custom feature extractor
        activation_fn=nn.LeakyReLU,  # Activation function
        optimizer_class=th.optim.Adam,  # Optimizer class
        optimizer_kwargs=dict(eps=2.5e-5),  # Learning rate for the optimizer
        ortho_init=True,  # Use orthogonal initialization
    )

    model_parameters = wandb.config.as_dict().copy()
    policy_kwargs.update(model_parameters["policy_kwargs"])

    model_parameters.pop("policy_kwargs")
    model_parameters.pop("number_of_experiments")
    model_parameters.pop("initial_learning_rate")
    model_parameters.pop("number_of_stacked_observations")

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

    # Create callback for logging, checkpoints and evaluation
    logging_callback = TensorboardCallback(verbose=0)

    eval_freq = 4_096
    eval_callback = EvalCallback(eval_env,
                                 eval_freq=eval_freq,
                                 n_eval_episodes=500,
                                 best_model_save_path=models_dir,
                                 deterministic=True,
                                 verbose=1)

    callbacks = CallbackList([logging_callback,
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
    mean_reward, _ = evaluate_policy(best_model, eval_env, n_eval_episodes=500, deterministic=True)
    run.log({"mean_reward": mean_reward})


# 2a: Default config and standard hyperparams
model_config = {
    # Policy and generic training settings
    "policy": "MultiInputPolicy",  # This is determined by what observation space you use.
    "device": "cuda",  # Performs network updates on cuda, if possible
    "seed": 42,
    "verbose": 1,

    # Hyperparameters, model specific
    # "learning_rate": 1e-6,
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
    "chromatogram_datapoints": 8192,
    "number_of_compounds": (10, 10),
    "number_of_segments": 1,
    "render_mode": 'rgb_array',
    "env_seed": None
}

extra_config = {
    # Additional settings
    "run_id": "ChomRL_Hyperpars_Sweep",
    "tb_log_name": f"RLChrom",
    "iterations": 100,
    "total_training_steps": 5_000_000,  # Total training interactions
    "number_of_parallel_envs": 24,  # Training in parallel envs. running on separate threads
    "evaluation_frequency": int(1024 * 100),  # How often to evaluate the agent
}

# 2b: Define the sweep search space, overwrites the default when sweeping
sweep_configuration = {
    "name":
        "ChomRL_Hyperpars_Sweep",
    "method":
        "grid",  # Can use grid, random or bayesian search
    "metric":
        {"goal": "maximize", "name": "mean_reward"},
    "parameters":
        {
            "number_of_stacked_observations":
                {"values": [10]},
            "number_of_experiments":
                {"values": [10]},
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
            "policy_kwargs":
                {
                    "parameters":
                        {
                            "net_arch":
                                {
                                    "parameters":
                                        {
                                            "pi":
                                                {"values": [
                                                    [256, 256],
                                                ]},
                                            "vf":
                                                {"values": [
                                                    [128, 128],
                                                ]},
                                        }
                                },
                            "log_std_init":
                                {"values": [-2]}
                        }
                }
        }
}

# 3: Start the sweep or single run
if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="PPO_Chromatography_Project")
    wandb.agent(sweep_id=sweep_id, function=main)
