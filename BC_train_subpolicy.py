import pathlib
from sys import path
from threading import local
from typing import Callable, Tuple
from heapq import heapify, heappush, heappop
import gym
from tqdm import tqdm
import pickle, os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from environments.twodimnav import TwoDimNav
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from imitation.algorithms import bc

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

from ExpertDemos import create_expert_demos
from utils import create_evaluate_policy_callback

def main():
    np.random.seed(0)
    grid_size = (10, 10)
    num_paths = 5000
    num_goals = 3
    start_goal_index = 0
    end_goal_index = 2
    expert_demos_filename = \
        f'data/expert_demos({grid_size},{num_paths},{num_goals},{start_goal_index},{end_goal_index}).pkl'
    
    # change if you want to guarantee the generation of new data or new
    # expert_demos object
    generate_new_data = False
    expert_demos_goal_1 = create_expert_demos(generate_new_data, expert_demos_filename,
                    grid_size, num_paths, num_goals, start_goal_index, end_goal_index)
    
    # obs = expert_demos_goal_1.obs
    # next_obs = expert_demos_goal_1.next_obs
    # dones = expert_demos_goal_1.dones
    # for i in range(obs.shape[0]):
    #     print(obs[i], next_obs[i], dones[i])
    # import pdb; pdb.set_trace()

    data_loader_goal_1 = DataLoader(expert_demos_goal_1, batch_size=32, shuffle=True)
    env = expert_demos_goal_1.env

    logs_dir = 'logs/'
    path_suffix = 'BC/subpolicy_approach'
    
    BC_logging_path = logs_dir + path_suffix
    print(f"All Tensorboards and logging are being written inside {BC_logging_path}/.")
    
    policy_save_path = 'policies/' + path_suffix
    print(f'Policies to be saved in {policy_save_path}/')
    # import pdb; pdb.set_trace()

    # Configure logger to write with tensorboard and to stdout 
    logger.configure(folder=BC_logging_path, format_strs=['tensorboard', 'stdout'])
    
    # Train BC on expert data.
    bc_trainer = bc.BC(env.observation_space, env.action_space, expert_data=data_loader_goal_1)
    # Create callback for tensorboard tracking of episodic reward
    writer = SummaryWriter(BC_logging_path)
    eval_callback = create_evaluate_policy_callback(writer, env)
    bc_trainer.train(n_epochs=100, log_interval=bc.BC.DEFAULT_BATCH_SIZE, 
        on_epoch_end=eval_callback)
    import pdb; pdb.set_trace()
    bc_trainer.save_policy(policy_save_path)
    # print(f'Num of expert demos: {len(data_loader.dataset)}')
    # TODO: add callback to bc_trainer

    print('Reconstructing policy...')
    bc_policy = bc.reconstruct_policy('policies/BC_Policy', 'cuda')
    print('Policy reconstructed!')
    # episode_rewards, episode_lengths = \
    #     evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10000, deterministic=True, return_episode_rewards=True)
    print('Evaluating policy')
    episode_rewards, episode_lengths = \
        evaluate_policy(bc_policy, env, n_eval_episodes=100000, deterministic=False, return_episode_rewards=True, render=False)
    

    print(f'Avg reward per episode: {np.mean(episode_rewards)}')
    print(f'Avg episode length: {np.mean(episode_lengths)}')


if __name__ == '__main__':
    main()