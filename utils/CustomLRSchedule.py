from typing import Callable

def custom_lr_schedule(initial_value: float, final_value: float, total_steps: int, decay_steps: int) -> Callable[
    [float], float]:
    """
    Custom learning rate schedule.

    :param total_steps: Total number of steps.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :param decay_steps: Number of steps in which to decay from initial to final lr.

    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        decay_rate = total_steps / decay_steps
        current_progress = (1 - progress_remaining)
        current_learning_rate = initial_value - (initial_value - final_value) * current_progress * decay_rate
        if current_learning_rate < final_value:
            current_learning_rate = final_value

        return current_learning_rate

    return func