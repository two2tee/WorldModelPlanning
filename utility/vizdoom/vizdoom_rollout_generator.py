from tqdm import tqdm
from utility.base_rollout_generator import BaseRolloutGenerator
from environment.environment_factory import get_environment


class RolloutGenerator(BaseRolloutGenerator):
    def __init__(self, config, data_output_dir):
        super().__init__(config,  data_output_dir)

    def _standard_rollout(self, thread, current_rollout, rollouts):
        action = 0
        environment = get_environment(self.config)
        _ = environment.reset()
        actions_rollout, states_rollout, reward_rollout, is_done_rollout = [], [], [], []

        progress_description = f"Data generation for {self.config['game']} | thread: {thread} | rollout: {current_rollout}/{rollouts}"
        for _ in tqdm(range(self.sequence_length + 1), desc=progress_description, position=thread - 1):
            obs, reward, done, info = environment.step(action)
            obs = self._compress_frame(obs, is_resize=True)
            actions_rollout.append(action)
            states_rollout.append(obs)
            reward_rollout.append(reward)
            is_done_rollout.append(done)
            environment.render()
            if done:
                environment.reset()
        environment.close()
        return actions_rollout, states_rollout, reward_rollout, is_done_rollout


