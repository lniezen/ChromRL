import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.solved_info_buffer = []
        self.variances_info_buffer = []
        #self.duration_info_buffer = []
        #self.avg_resolution_info_buffer = []
        #self.critical_resolution_info_buffer = []
        #self.num_experiments_info_buffer = []

    def _on_step(self) -> bool:
        # Get the `dones` flag and `infos`
        done_flags = self.locals['dones']
        infos = self.locals['infos']

        # Check if the episode has ended
        for i in range(len(done_flags)):
            if done_flags[i]:
                # Extract and store relevant information from `infos` at episode end
                if 'Solved' in infos[i]:
                    self.solved_info_buffer.append(infos[i]['Solved'])
                if 'Mean_variance' in infos[i]:
                    self.variances_info_buffer.append(infos[i]['Mean_variance'])
                #if 'Duration' in infos[i]:
                #    self.duration_info_buffer.append(infos[i]['Duration'])
                #if 'Average_resolution' in infos[i]:
                #    self.avg_resolution_info_buffer.append(infos[i]['Average_resolution'])
                #if 'Critical_resolution' in infos[i]:
                #    self.critical_resolution_info_buffer.append(infos[i]['Critical_resolution'])
                #if 'Number_of_experiments' in infos[i]:
                #    self.num_experiments_info_buffer.append(infos[i]['Number_of_experiments'])

        return True

    def _on_rollout_end(self):
        # Log mean of collected infos if there is data in the buffers
        if len(self.solved_info_buffer) > 0:
            mean_solved = np.mean(self.solved_info_buffer)
            self.logger.record("rollout/mean_solved", mean_solved)
        if len(self.variances_info_buffer) > 0:
            mean_variance = np.mean(self.variances_info_buffer)
            self.logger.record("rollout/mean_variance", mean_variance)
        #if len(self.duration_info_buffer) > 0:
        #    mean_duration = np.mean(self.duration_info_buffer)
        #    self.logger.record("mean_duration", mean_duration)
        #if len(self.avg_resolution_info_buffer) > 0:
        #    mean_rs_avg = np.mean(self.avg_resolution_info_buffer)
        #    self.logger.record("mean_rs_avg", mean_rs_avg)
        #if len(self.critical_resolution_info_buffer) > 0:
        #    mean_rs_crit = np.mean(self.critical_resolution_info_buffer)
        #    self.logger.record("mean_rs_crit", mean_rs_crit)
        #if len(self.num_experiments_info_buffer) > 0:
        #    mean_num_experiments = np.mean(self.num_experiments_info_buffer)
        #    self.logger.record("mean_number_of_experiments", mean_num_experiments)

        # Clear the buffers after logging
        self.solved_info_buffer.clear()
        self.variances_info_buffer.clear()
        # self.duration_info_buffer.clear()
        # self.avg_resolution_info_buffer.clear()
        # self.critical_resolution_info_buffer.clear()
        # self.num_experiments_info_buffer.clear()

        # Dump logs at the end of the rollout
        self.logger.dump(self.num_timesteps)