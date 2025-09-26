from utils.env_utility_funcs import *
import os
import numpy as np
import gymnasium as gym
import pandas as pd


class Chromatography(gym.Env):
    """Chromatographic environment."""

    def __init__(
            self,
            time: float,
            num_actions: int,
            num_experiments: int,
            num_compounds: tuple[int, int] = (10, 20),
            resolution_target: float = 1.5,
            time_target: float = 20.0,
            chromatogram_size: int = 8192,
            initial_action: np.ndarray = None,
            reward_info: bool = False,
            seed: int = None,
            render_fn: callable = None,
            save_path: str = None,
            solvable_mixtures: dict = None,
            dtype: np.dtype = np.float32,
    ) -> None:

        super(Chromatography, self).__init__()

        self.mixture_generator = MixtureGenerator(
            *num_compounds, dtype=dtype)

        self.time = Time(
            start=0.0,
            end=time,
            delta=time / chromatogram_size,
            dtype=dtype)

        self.program = MobilePhaseProgram(
            time=self.time,
            num_segments=(num_actions - 1),
            dtype=dtype)

        self.crf = ChromatographicResponseFunction(
            resolution_target=resolution_target,
            time_target=time_target,
            dtype=dtype)

        self.num_experiments = num_experiments
        self.max_exp_time = time_target

        self._seed = seed
        self.previous_seed = seed

        self._solvable_mixtures = solvable_mixtures
        self.num_actions = num_actions
        self._dtype = dtype
        self._save_path = save_path

        if render_fn is None:
            self._render_fn = _plot_separation
        else:
            self._render_fn = render_fn

        if initial_action is None:
            # phi_start at 0.3 (-0.4) to make sure all, or the very majority,
            # of compounds elute within the maximum time (usually between 
            # 20-30 minutes).
            self.initial_action = np.linspace(
                -0.4, 1.0, num_actions, dtype=self._dtype)
        else:
            self.initial_action = np.zeros(
                shape=(num_actions,), dtype=self._dtype)
            self.initial_action += initial_action

        self.observation_space = gym.spaces.Dict(
            {
                'chromatogram': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.time.array.shape,
                    dtype=self._dtype),
                'phi_target': gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=self.initial_action.shape,
                    dtype=self._dtype),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.initial_action.shape,
            dtype=self._dtype)

    def reset(self, **kwargs) -> tuple:
        self.step_counter = 0

        # SB3 resets an additional time, this circumvents the seed incrementing incorrectly
        if self._seed != self.previous_seed and self._seed is not None:
            self._seed = self.previous_seed
        if self._seed is not None:
            self._seed += 1

        self.mixture = self.mixture_generator.generate(seed=self._seed)
        # Ensure that the sample is always selected from a known solvable sample set (e.g. sample_set_10p.npy) 
        if self._seed is not None:
            if self._seed > len(self._solvable_mixtures):
                self._seed = self._seed - len(self._solvable_mixtures)
            if self._seed <= len(self._solvable_mixtures):
                self.mixture = self._solvable_mixtures[self._seed - 1]

        scaled_action = self.initial_action / 2. + 0.5
        self.program.set(scaled_action)
        separation = self.program.run(self.mixture, seed=self._seed)

        chromatogram = separation.observation(self._seed)

        observation = {
            'chromatogram': chromatogram,
            'phi_target': self.initial_action,
        }

        info = {}

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        self.step_counter += 1
        self.previous_seed = self._seed

        action = action.astype(self._dtype)
        scaled_action = action / 2. + 0.5

        self.program.set(scaled_action)
        separation = self.program.run(self.mixture, seed=self._seed)
        chromatogram = separation.observation(self._seed)

        observation = {
            'chromatogram': chromatogram,
            'phi_target': action,
        }

        result = self.crf(separation)
        solved = 0.0
        reward = 0.0
        terminal = truncated = False
        # Attribute a reward of 1.0 if the sample is solved (all peaks achieving required resolution and eluting within the time threshold)
        if result.reward == 1.0:
            solved = 1.0
            reward = 1.0
            terminal = True
        # Terminate the episode if the experimental budget is fully used up.
        elif self.step_counter > self.num_experiments:
            truncated = True

        # Additional info for logging
        info = {
            'Solved': solved,
        }

        return observation, reward, terminal, truncated, info

    def render(self) -> np.ndarray:
        # Not used
        return self.separation.observation(), self.result.reward


def _plot_separation(
        separation: Separation,
        program: MobilePhaseProgram,
        result: float,
        xlabel: str = 'Time (min)',
        y1label: str = 'Absorbance',
        y2label: str = '$\phi$',
        y1lim: tuple[float, float] = (-0.10, 1.50),
        y2lim: tuple[float, float] = (-0.05, 1.05),
        save_path: str = None,
        save_excel_path: str = None  # Add parameter for Excel saving path
) -> None:
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 3))

    blue, orange, green = 'C0', 'C1', 'C2'

    x = separation.time.array
    y = separation.observation()

    ax1.set_title(
        r'$\bar{{R}}_{{avg}}$ = {:.7f}'.format(result.reward),
        fontsize=10,
        color=green)

    ax1.plot(x, y, color=blue, zorder=2)
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel(y1label, color=blue, fontsize=10)
    # ax1.set_ylim(*y1lim)

    ax2 = ax1.twinx()

    ax2.plot(x, program.phi, color=orange, zorder=1)
    ax2.set_ylabel(y2label, color=orange, fontsize=12)
    ax2.set_ylim(*y2lim)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    indices = np.argsort(separation.loc)

    locs = separation.loc[indices]

    for index, loc in zip(indices, locs):
        xtext, ytext = loc, y[np.argmin(np.abs(x - loc))]
        ax1.scatter(xtext, ytext + 0.05, s=10)
    #  ax1.text(xtext, min(ytext, y1lim[1]), f'{index}', fontsize=8, va='bottom', ha='center')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)

    if save_excel_path:
        # Flatten arrays to 1D if they are not already
        x_flat = np.ravel(x)
        y_flat = np.ravel(y)
        phi_flat = np.ravel(program.phi)
        locs_flat = np.ravel(separation.loc)

        # Determine the maximum length
        max_length = max(len(x_flat), len(y_flat), len(phi_flat), len(locs_flat))

        # Pad arrays to make them all the same length
        x_flat = np.pad(x_flat, (0, max_length - len(x_flat)), constant_values=np.nan)
        y_flat = np.pad(y_flat, (0, max_length - len(y_flat)), constant_values=np.nan)
        phi_flat = np.pad(phi_flat, (0, max_length - len(phi_flat)), constant_values=np.nan)
        locs_flat = np.pad(locs_flat, (0, max_length - len(locs_flat)), constant_values=np.nan)

        data = {
            'Time (min)': x_flat,
            'Absorbance': y_flat,
            'Phi': phi_flat,
            'Locs': locs_flat  # Add locs as a new column
        }

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(save_excel_path), exist_ok=True)
        df.to_excel(save_excel_path, index=False)


    plt.clf()
    plt.close()
