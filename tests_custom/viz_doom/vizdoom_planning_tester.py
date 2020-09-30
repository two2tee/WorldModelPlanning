import time
from tests_custom.base_planning_tester import BasePlanningTester, TEST_NAME, ELITES, CUSTOM_SEED, ACTION_HISTORY


class VizDoomPlanningTester(BasePlanningTester):
    def __init__(self, config, vae, mdrnn, preprocessor, planning_agent):
        super().__init__(config, vae, mdrnn, preprocessor, planning_agent)

    def get_test_functions(self):
        return {  # (test_func, args)
            "random_takecover": (self._random_takecover, {CUSTOM_SEED: None})
        }

    def _get_trial_results_dto(self, args):
        base_dto = super(VizDoomPlanningTester, self)._get_trial_results_dto(args)
        # TODO figure out what additional results to store
        return base_dto

    def _update_trial_results(self, trial_results_dto , total_reward):
        trial_results_dto['max_reward'] = total_reward if total_reward > trial_results_dto['max_reward'] else trial_results_dto['max_reward']

    def _print_trial_results(self, trial, seed, elapsed_time, total_reward, steps_ran, trial_results_dto):
        test_name = trial_results_dto['test_name']
        test_success = trial_results_dto['test_success']

        trial_str = '' if trial is None else f'Planning trial: {trial}'
        elapsed_time_str = '' if elapsed_time is None else f'Elapsed_time: {round(elapsed_time, 0)}'
        success_str = f'Test success: {test_success}' \
            if test_success else f'Test success: {test_success}\n'
        print(
            f'\n\nRESULTS FOR: {test_name}\n{trial_str} | {elapsed_time_str}\n'
            f'Total reward: {round(total_reward, 2)} | Max reward: {trial_results_dto["max_reward"]} |  Steps on exit: {steps_ran}\n'
            f'{success_str}')

    # TEST METHODS #######################################

    def _random_takecover(self, args):
        args[TEST_NAME] = '----- Forward planning test ------'
        return self._run_plan_or_replay(args=args)
    ##

    def _run_trial(self, trial_i, args, seed):
        environment = self._get_environment()
        current_state = environment.reset(seed)
        seed = environment.environment.seed
        _ = self.simulated_environment.reset()

        latent_state, _ = self._encode_state(current_state)
        hidden_state = self.simulated_environment.get_hidden_zeros_state()

        trial_results_dto = self._get_trial_results_dto(args)
        elites = []
        action_history = []
        start_time = time.time()
        total_reward = 0
        steps_ran = 0
        elapsed_time = 0
        is_done = False

        while not is_done:
            action, step_elites = self._search_action(latent_state, hidden_state)
            elites.append(step_elites)

            if self.is_render_dream:
                self._step_sequence_in_dream(self.planning_agent.current_elite.action_sequence, current_state, hidden_state)

            current_state, reward, is_done, simulated_reward, simulated_is_done, latent_state, hidden_state = self._step(action, hidden_state, environment)

            action_history.append(action)
            total_reward += reward
            steps_ran += 1
            elapsed_time = time.time() - start_time
            self._update_trial_results(trial_results_dto, reward)

        self._print_trial_results(trial_i, seed, elapsed_time, total_reward, steps_ran, trial_results_dto)
        environment.close()
        return elites, action_history, total_reward, trial_results_dto['max_reward'], seed

    def _replay_planning_test(self, args):
        actions = args[ACTION_HISTORY]
        elites = args[ELITES]
        seed = args[CUSTOM_SEED]
        environment = self._get_environment()
        _ = environment.reset(seed=seed)
        self.simulated_environment.reset()
        hidden_state = self.simulated_environment.get_hidden_zeros_state()
        trial_results_dto = self._get_trial_results_dto(args)

        steps_ran = 0
        total_reward = 0
        for i, action in enumerate(actions):
            current_state, reward, is_done, simulated_reward, simulated_is_done, latent_state, hidden_state = self._step(action, hidden_state)
            steps_ran += 1
            total_reward += reward
            self._update_trial_results(trial_results_dto, reward)

            if self.is_render_dream:
                self._step_sequence_in_dream(elites[i][-1][2], current_state, hidden_state)

        self._print_trial_results(None, None, total_reward, steps_ran, trial_results_dto)
        environment.close()

        return actions, total_reward


