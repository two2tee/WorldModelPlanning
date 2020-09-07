""" Main entrypoint for running the solution """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import json
from vae.vae import VAE
from mdrnn.mdrnn import MDRNN
from vae.vae_trainer import VaeTrainer
from colorama import init as colorama_init
from utility.preprocessor import Preprocessor
from mdrnn.iterative_trainer import IterativeTrainer
from tuning.ntbea_wrapper import PlanningNTBEAWrapper
from utility.rollout_handling.rollout_generator_factory import get_rollout_generator
from tests_custom.test_suite_factory import get_model_tester, get_planning_tester
from environment.environment_factory import get_environment
from mdrnn.mdrnn_trainer import MDRNNTrainer as MDRNNTrainer
from planning.simulation.simulated_planning_controller import SimulatedPlanningController
from planning.agent_factory import get_planning_agent
colorama_init()


class Main:
    def __init__(self, config):
        self.config = config
        self.frame_preprocessor = Preprocessor(self.config['preprocessor'])
        self.mdrnn_trainer = MDRNNTrainer(self.config, self.frame_preprocessor)
        self.environment = get_environment(config)  # Set environment


    def generate_data(self):
        data_handler = get_rollout_generator(self.config)
        data_handler.generate_rollouts()

    def train_or_reload_vae(self):
        vae_trainer = VaeTrainer(self.config, self.frame_preprocessor)
        vae = VAE(self.config)
        return vae_trainer.train(vae) if config["is_train_vae"] else vae_trainer.reload_model(vae)

    def train_or_reload_mdrnn(self):
        mdrnn = MDRNN(num_actions=self.environment.action_sampler.num_actions,
                      latent_size=self.config['latent_size'],
                      num_gaussians=self.config['mdrnn']['num_gaussians'],
                      num_hidden_units=self.config['mdrnn']['hidden_units'])

        if config["is_iterative_train_mdrnn"]:
            return self._iterative_mdrnn_training(mdrnn)
        elif self.config["is_train_mdrnn"]:
            return self._standard_mdrnn_training(mdrnn)
        else:
            return self.mdrnn_trainer.reload_model(mdrnn)

    def _standard_mdrnn_training(self, mdrnn):
        trained_mdrnn, _ = self.mdrnn_trainer.train(vae, mdrnn)
        return trained_mdrnn

    def _iterative_mdrnn_training(self, mdrnn):
        planning_agent = get_planning_agent(self.config)
        iterative_trainer = IterativeTrainer(config, planning_agent, main.mdrnn_trainer)
        iterative_trainer.train()
        self.mdrnn_trainer.reload_model(mdrnn)
        return

    def run_ntbea_tuning(self, vae, mdrnn):
        agent = get_planning_agent(self.config)
        planning_tester = get_planning_tester(self.config, vae, mdrnn, self.frame_preprocessor, self.environment, agent)
        ntbea_tuner = PlanningNTBEAWrapper(self.config, planning_tester)
        ntbea_tuner.run_ntbea()

    def run_model_tests(self, vae, mdrnn):
        model_tester = get_model_tester(self.config, vae, mdrnn, self.frame_preprocessor, self.environment)
        model_tester.run_tests()

    def run_planning_tests(self, vae, mdrnn):
        agent = get_planning_agent(self.config)
        planning_tester = get_planning_tester(self.config, vae, mdrnn, self.frame_preprocessor, self.environment, agent)
        planning_tester.run_tests()

    def play_game(self, vae, mdrnn):
        print('---- START PLAYING ----')
        agent = get_planning_agent(self.config)
        game_controller = SimulatedPlanningController(config, main.frame_preprocessor, vae, mdrnn)
        num_games, average_steps, average_reward = 10, 0, 0
        for i in range(num_games):
            total_steps, total_reward, total_simulated_reward = game_controller.play_game(agent, self.environment)
            average_steps += total_steps
            average_reward += total_reward
            print(f"Game: {i} | Total steps: {total_steps} | Total reward: {total_reward}")
            print(f"Average steps: {average_steps / num_games} | Average reward: {average_reward / num_games}")

# MAIN LOOP ###########################
with open('config.json') as config_file:
    config = json.load(config_file)

if __name__ == '__main__':

    session_name = config['experiment_name']
    print(f'Session: {session_name}')
    main = Main(config)

    if config["is_generate_data"]:
        main.generate_data()

    vae = main.train_or_reload_vae()
    mdrnn = main.train_or_reload_mdrnn()

    if config['test_suite']["is_run_model_tests"]:
        main.run_model_tests(vae, mdrnn)

    if config['is_ntbea_param_tune']:
        main.run_ntbea_tuning(vae, mdrnn)

    if config['test_suite']["is_run_planning_tests"]:
        main.run_planning_tests(vae, mdrnn)

    if config["is_play"]:
        main.play_game(vae, mdrnn)
