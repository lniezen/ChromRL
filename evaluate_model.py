import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from utils.env_utility_funcs import plot_contour
from envs.env_chromatogram_obs import Chromatography
# from envs.env_peak_obs import Chromatography
# from envs.env_mixed_obs import Chromatography


def evaluate_trained_agent(model, env, sample_data, plot_dir, save_plots):
    """
    Evaluate the policy by running multiple episodes (samples)
    and computing the mean total reward. (fraction of solved samples)
    """

    n_eval_episodes = len(sample_data)
    episode_rewards = []

    phi_0_grid = np.linspace(0, 1.0, 100)
    phi_1_grid = np.linspace(0, 1.0, 100)

    for sample in range(n_eval_episodes):
        t = 0

        obs = env.reset()
        done = False
        total_reward = 0.0  # accumulate reward over the whole episode

        initial_action = [0.2, 1.0]
        used_values = [(initial_action, [0, 20.0])]
        score_true = sample_data[sample]["Response_Surface"]
        while not done:
            t += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)

            total_reward += reward  # accumulate step reward

            scaled_action = action / 2 + 0.5
            scaled_action = np.ravel(scaled_action).tolist()
            used_values.append((scaled_action, [0, 20.0]))
            if terminated:
                done = True

        episode_rewards.append(total_reward)
        print(f"Episode: {sample+1}/{n_eval_episodes}, Mean Solved: {np.mean(episode_rewards):.3f}")

        if save_plots:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plot_contour(
                ax,
                phi_0_grid, phi_1_grid,
                surface=np.asarray(score_true, dtype=float),
                title=["True Score"],
                best_phi_points=scaled_action,
                best_time_points=[0, 20.0],
                optimal_phi_ini=None,
                optimal_phi_fin=None,
                used_values=used_values,
                trial_num=t,
                cmap="plasma",
            )
            ax.legend(loc="upper right")  # e.g. the last one
            plt.show()
            plt.savefig(f"{plot_dir}/summary_sample_{sample + 1}_trial_{t}_.png", dpi=600)
            plt.close()

    mean_reward = np.mean(episode_rewards)
    return mean_reward


def main():
    run_id = "envj4ofr"
    models_dir = f"models/{run_id}/"

    save_plots = False
    plot_dir = f"action_trajectories/{run_id}/"
    os.makedirs(plot_dir, exist_ok=True)

    # Environment settings, must match with the training env.
    env_config = {
    "maximum_experiment_duration": 20.0,
    "number_of_stacked_observations": 10,
    "time_target": 20.0,
    "chromatogram_datapoints": 8192,
    "number_of_compounds": (15, 15),
    "number_of_segments": 1,
    }

    # Load sample mixtures
    filename = "sample_sets/evaluation_samples_15p.npy"
    sample_data = np.load(filename, allow_pickle=True)
    sample_mixtures = [entry["Mixture"] for entry in sample_data]
    cfg = env_config

    eval_env = make_vec_env(
        Chromatography,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={
            "time": cfg["maximum_experiment_duration"],
            "time_target": cfg["time_target"],
            "chromatogram_size": cfg["chromatogram_datapoints"],
            "num_experiments": 20,
            "num_compounds": cfg["number_of_compounds"],
            "num_actions": cfg["number_of_segments"] + 1,
            "seed": 0,
            "solvable_mixtures": sample_mixtures,
        },
    )
    eval_env = VecFrameStack(eval_env, n_stack=cfg["number_of_stacked_observations"], channels_order="last")

    # Evaluate model
    best_model_path = os.path.join(models_dir, "best_model")
    best_model = PPO.load(best_model_path)
    evaluate_trained_agent(best_model, eval_env, sample_data, plot_dir, save_plots)

if __name__ == '__main__':
    main()
