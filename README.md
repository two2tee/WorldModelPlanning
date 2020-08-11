# Evolutionary Planning on Learned World Model - Thesis Implementation

This is the codebase for the Evolutionary Planning on Learned World Model thesis written by Thor V.A.N. Olesen and Dennis Thinh Tan Nguyen. 

The goal of this work is to use **evolution to do online-planning on a learned model of the world** and and is inspired by the Paper: Ha and Schmidhuber, "*World Models*", 2018.https://doi.org/10.5281/zenodo.1207631.

Similarly to the paper we use a Convolutional Variational Auto Encoder (ConvVAE) to learn a representation of the world and a Mixture Density Recurrent Neural Network (MDRNN) to learn the dynamics of the world. 

While the paper uses a simple linear policy network (controller) to produce actions, we use the **Rolling Horizon Evolutionary Algorithm (RHEA)** in conjunction with the world model as a forward model to perform online planning. More information about RHEA can found in the paper:  Raluca D. Gaina and Sam Devlin, Simon M. Lucas, and Diego Perez-Liebana, "*Rolling Horizon Evolutionary Algorithms for General Video Game Playing*",  2020. https://arxiv.org/pdf/2003.12331.pdf

Finally, we have focused our attention on planning in the single-player real-time game  ’Car-Racing’  by Open AI https://gym.openai.com/envs/CarRacing-v0/

## Prerequisites
The system is written in Python 3.7 and utilizes PyTorch 1.5. Please refer to https://pytorch.org/ for PyTorch installation details. The rest of the dependencies are located in *requirements.txt*

```batch
pip3 install -r requirements.txt
```

## Hyperparameters
The hyperparameters are stored in a json file and can be in *config.json*.
The next few sections only showcases the essentials but you are welcome to play around with the other parameters.

## Generate Data
To generate new rollouts set the following hyperparamers in the config file:
1. ``is_generate_data: true``
2. ``data_generator: {rollouts: 10000 }``

By default, a random policy is used but a good policy can be enabled with ``is_ha_agent_driver: true``

## Models
This sections explains how the VAE and MDRNN can be configured for training or reloading.

### Training VAE
To train a new model VAE set the following hyperparamers in the config file:
1. ``experiment_name: "SOME_NAME_YOU_DEFINE"``
2. ``is_train_vae: true``
3. ``latent_size: 64``
4. ```  
    vae_trainer:{
        "max_epochs": 20,
        "batch_size": 35,
        "learning_rate": 0.0001
    }
   ```
6.  Run

#### Reloading VAE
To reload a VAE, set the parameters accordingly:
1. ``is_train_vae: false``
2. ``experiment_name: "SOME_NAME_YOU_DEFINE"``

### Training MDRNN
To train a new model VAE set the following hyperparamers in the config file:
1. ``experiment_name: "SOME_NAME_YOU_DEFINE"``
2. ``is_train_mdrnn: true``
3. ``latent_size: 64``
4. ```  
    mdrnn_trainer:{
        max_epochs": 20,
        "learning_rate": 0.001,
        "sequence_length": 500,
    }
   ```
   
5.  ``"mdrnn":{ "hidden_units": 512 } ``
6.  Run

#### Reloading MDRNN
To reload a VAE, set the parameters accordingly:
1. ``is_train_mdrnn: false``
2. ``experiment_name: "SOME_NAME_YOU_DEFINE"``
3. **Ensure** that the ``hidden_units`` and ``latent_size`` are set **exactly** to the values used to train the MDRNN.

## Planning Algorithms
The planning algorithms available are: 
1. **RHEA - Rolling Horizon Evolutionary Algorithm**
2. **RMHC - Random Mutation Hill Climbing**
3. **MCTS - Monte Carlo Tree Search**

To choose an agent, set the parameter to either **RHEA, RMHC** or **MCTS**
+ ``"planning: { "planning_agent": RHEA }"``

All other agent parameters can be played with as accordingly:
```
"planning": {
        "planning_agent": "RHEA",
           "rolling_horizon": {
            "population_size": 4,
            "horizon": 10,
            "max_generations": 15,
            "is_shift_buffer": false
        },
        "random_mutation_hill_climb": {
            "horizon": 50,
            "max_generations": 15,
            "is_shift_buffer": false,
        },
        "monte_carlo_tree_search": {
            "max_rollouts": 50,
            "rollout_length": 20,
            "temperature": 1.41,
            "is_discrete_delta": true
        }
```

## NTBEA Parameter tuning
**N-Tuple Bandit Evolutionary Algorithm** (NTBEA) has been implemented to perform parameter tuning on the planning parameters.
The implementation is based on https://github.com/bam4d/NTBEA and more information about
NTBEA can be read in the paper Lucas, Liu, Perez-Liebana, "*The N-Tuple Bandit Evolutionary Algorithm for Game Agent Optimisation*", 2018.https://arxiv.org/abs/1802.05991

To run NTBEA tuning set the following configuration:
1. ``is_ntbea_param_tune: True``
2. Select World Model: ``"experiment_name": "NAME OF WORLD MODEL"``
3. Select an agent: ``"planning: { "planning_agent": RHEA }"``
4. Run

## Planning Benchmarks
To run planning benchmarks of the agents + world model set the following parameters:
1. Select World Model: ``"experiment_name": "NAME OF WORLD MODEL"``
2. Select an agent: ``"planning: { "planning_agent": RHEA }"``
3. ```
    "test_suite": {
        "is_run_model_tests": false,
        "is_run_planning_tests":  true,
        "is_reload_planning_session": false,
        "trials": 3,
    }
    ```

#### Replay benchmark session
To replay a benchmark session set the following parameters and run:
```
    "test_suite": {
        "is_run_model_tests": false,
        "is_run_planning_tests":  true,
        "is_reload_planning_session": true,
        "trials": 3,
        "planning_session_to_load": "name of file without '.pickle'"
    },
```

## Live Play
To deploy the agent on a random game, set the following parameters:
1. Select World Model: ``"experiment_name": "NAME OF WORLD MODEL"``
2. Select an agent: ``"planning: { "planning_agent": RHEA }"``
3.  ``"is_play": true``
4.  Run

### Play in Dream
To enable playing in dream, use the above parameters but enable: 
+ ``"is_dream_play": true``

### Manual Play
To manually control the car, use the above parameters but enable:
+ ``"is_manual_control": true``

NB: You can likewise drive in the dream by enabling ``is_dream_play``

## Existing Models
The solution comes with a set of pretrained models that can be used. The best performing model is Model L.
1. To select a world model go to config and set the experiment name attribute with a world model name in the world: ``"experiment_name": "SELECTED_WORLD_MODEL"``

2. In the config set latent_size and hidden_units accordingly to the parameters below eg:
    
    + Example:
        ```
        experiment_name: World_Model_L
        latent_size    : 64
        hidden_units   : 512 // MDRNN units
        ```

3. Run the program

**Models and their parameters**
```

World Model Name | Parameters (epochs, sequence, latent_size, mdrnn_hidden_units, rollouts used for training)

World_Model_A    |  20_epoch 64Seq  32Latent 256Hidden  10k_Brownian_policy
World_Model_B    |  20_epoch 64Seq  32Latent 256Hidden  10k_Ha_policy
World_Model_C    |  20_epoch 64Seq  32Latent 256Hidden  20k_Ha_Brownian_policy
World_Model_D    |  20_epoch 64Seq  32Latent 256Hidden  10k_Brownian_5k_passive_policy
World_Model_E    |  20_epoch 64Seq  32Latent 256Hidden  25k_Ha_Brownian_passive_policy
World_Model_F    |  20_epoch 64Seq  32Latent 256Hidden  5kHa_5kBrownian_policy
World_Model_G    |  20_epoch 64Seq  64Latent 256Hidden  20k_Ha_Brownian_policy
World_Model_H    |  20_epoch 64Seq  32Latent 512Hidden  20k_Ha_Brownian_policy
World_Model_I    |  20_epoch 64Seq  64Latent 512Hidden  20k_Ha_Brownian_policy
World_Model_J    |  20_epoch 64Seq  64Latent 512Hidden  5kHa_5kBrownian_corners_policy
World_Model_K    |  20_epoch 64Seq  64Latent 512Hidden  20k_Ha_Brownian_10k_HaBrownRight_corners_policy
World_Model_L    |  20_epoch 500Seq 64Latent 512Hidden  20k_Ha_Brownian_policy
World_Model_M    |  20_epoch 500Seq 64Latent 512Hidden  20k_Brownian_policy
World_Model_N    |  20_epoch 500Seq 64Latent 1024Hidden 20k_Ha_Brownian_policy
```
## Copyright 
Copyright (c) 2020, - All Rights Reserved -
All files are part of *the Evolutionary Planning on a Learned World Model* thesis.
Unauthorized copying or distribution of the project, via any medium is strictly prohibited without the consensus of the authors.

**Authors**: 
+ **Thor V.A.N. Olesen** <thorolesen@gmail.com>
+ **Dennis T.T. Nguyen** <dennisnguyen3000@yahoo.dk>.