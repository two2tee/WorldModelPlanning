import time

from tests.base_planning_tester import BasePlanningTester, TEST_NAME, ELITES, CUSTOM_SEED, ACTION_HISTORY

class VizDoomPlanningTester(BasePlanningTester):
    def __init__(self, config, vae, mdrnn, preprocessor, environment, planning_agent):
        super().__init__(config, vae, mdrnn, preprocessor, environment, planning_agent)

    def get_test_functions(self):
        return {  # (test_func, args)
            "random_takecover": (self._random_takecover, {CUSTOM_SEED: None})
        }

    def _get_trial_results_dto(self, args):
        return NotImplemented

    def _update_trial_results(self, trial_results_dto, reward, total_reward, steps_ran):
        return NotImplemented

    def _print_trial_results(self, trial, elapsed_time, total_reward, steps_ran, trial_results_dto):
        return NotImplemented



    # TEST METHODS #######################################

    def _random_takecover(self, args):
        args[TEST_NAME] = '----- Forward planning test ------'
        return self._run_plan_or_replay(args=args)
    ##

    def _run_plan_or_replay(self, args):
        print(args[TEST_NAME])
        if ACTION_HISTORY in args:
            return self._replay_planning_test(args)
        else:
            return self._run_planning_test(args)

    def _run_planning_test(self, args):
        trial_actions = []
        trial_rewards = []
        trial_max_rewards = []
        trial_elites = []

        seed = args[CUSTOM_SEED]

        for i in range(self.trials):
            current_state = self.environment.reset(seed)
            seed = self.environment.seed
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

            for step in range(args['optimal_steps'] + 75):
                if self.config['planning']['planning_agent'] == "MCTS":
                    action = self.planning_agent.search(self.simulated_environment, latent_state, hidden_state)
                else:
                    action, step_elites = self.planning_agent.search(self.simulated_environment, latent_state, hidden_state)
                    elites.append(step_elites)

                if self.is_render_dream:
                    self._step_sequence_in_dream(self.planning_agent.current_elite.action_sequence, current_state, hidden_state)

                current_state, reward, is_done, simulated_reward, simulated_is_done, latent_state, hidden_state = self._step(action, hidden_state)

                action_history.append(action)
                total_reward += reward
                steps_ran += 1
                elapsed_time = time.time() - start_time
                self._update_trial_results(trial_results_dto, reward, total_reward, steps_ran)

            trial_elites.append(elites)
            trial_actions.append(action_history)
            trial_rewards.append(total_reward)
            trial_max_rewards.append(trial_results_dto['max_reward'])

            self._print_trial_results(i, elapsed_time, total_reward, steps_ran, trial_results_dto)

        return trial_actions, trial_rewards, trial_elites, trial_max_rewards, seed

    def _replay_planning_test(self, args):
        actions = args[ACTION_HISTORY]
        elites = args[ELITES]
        seed = args[CUSTOM_SEED]

        _ = self.environment.reset(seed=seed)
        self.simulated_environment.reset()
        hidden_state = self.simulated_environment.get_hidden_zeros_state()
        trial_results_dto = self._get_trial_results_dto(args)

        steps_ran = 0
        total_reward = 0
        for i, action in enumerate(actions):
            current_state, reward, is_done, simulated_reward, simulated_is_done, latent_state, hidden_state = self._step(action, hidden_state)
            steps_ran += 1
            total_reward += reward
            self._update_trial_results(trial_results_dto, reward, total_reward, steps_ran)

            if self.is_render_dream:
                self._step_sequence_in_dream(elites[i][-1][2], current_state, hidden_state)

        self._print_trial_results(None, None, total_reward, steps_ran, trial_results_dto)

        return actions, total_reward

    def _step_sequence_in_dream(self, actions, current_state, hidden):
        latent, _ = self._encode_state(current_state)
        total_reward = 0

        for action in actions:
            latent, simulated_reward, simulated_is_done, hidden = self.simulated_environment.step(action, hidden, latent,
                                                                                                  is_simulation_real_environment=True)
            total_reward += simulated_reward
            self.simulated_environment.render()
        return total_reward