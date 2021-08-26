import pathlib
from threading import local
from typing import Callable, Tuple
import gym
from tqdm import tqdm
import pickle, os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util
from ExpertDemos import ExpertDemos

# TODO: move to above
from stable_baselines3.common.vec_env import VecEnv

from ExpertDemos import create_expert_demos
from utils import create_bc_policy_callback

def main():
    np.random.seed(0)
    grid_size = (10, 10)
    num_paths = 5000
    num_goals = 5
    start_goal_index = 0
    end_goal_index = num_goals - 1
    expert_demos_filename = (f'data/expert_demos({grid_size},{num_paths},{num_goals},'
            f'{start_goal_index},{end_goal_index}).pkl')
    
    # change if you want to guarantee the generation of new data
    generate_new_data = False
    expert_demos = create_expert_demos(generate_new_data, expert_demos_filename,
                    grid_size, num_paths, num_goals, start_goal_index, end_goal_index)
    
    # TODO: remove
    # expert_demos = ExpertDemos(size=(10, 10), num_paths=5000, num_goals=3,
    #     goal_to_start_from=2, goal_to_end_at=2)
    # obs = expert_demos.obs
    # acts = expert_demos.acts
    # next_obs = expert_demos.next_obs
    # dones = expert_demos.dones

    # # TODO: remove
    # for i in range(obs.shape[0]):
    #     print(obs[i], next_obs[i], dones[i])
    # import pdb; pdb.set_trace()

    data_loader = DataLoader(expert_demos, batch_size=32, shuffle=True)
    env = expert_demos.env

    logs_dir = 'logs/'
    path_suffix = 'BC/onepolicy_approach/'
    
    BC_logging_path = logs_dir + path_suffix
    print(f"All Tensorboards and logging are being written inside {BC_logging_path}.")
    
    policy_save_path = 'policies/' + path_suffix + 'policy'
    print(f'Policies to be saved in {policy_save_path}')
    # import pdb; pdb.set_trace()

    # Configure logger to write with tensorboard and to stdout 
    logger.configure(folder=BC_logging_path, format_strs=['tensorboard', 'stdout'])
    
    # Train BC on expert data.
    bc_trainer = bc.BC(env.observation_space, env.action_space, expert_data=data_loader)
    print(f'Using {bc_trainer.device} device')
    # Create callback for tensorboard tracking of episodic reward
    writer = SummaryWriter(BC_logging_path)
    eval_callback = create_bc_policy_callback(writer, env, performance_difference=1,
        policy_save_path=policy_save_path, n_eval_episodes=500)
    bc_trainer.train(n_epochs=200, log_interval=bc.BC.DEFAULT_BATCH_SIZE, 
        on_epoch_end=eval_callback)
    # print(f'Num of expert demos: {len(data_loader.dataset)}')

    # print('Reconstructing policy...')
    # bc_policy = bc.reconstruct_policy(policy_save_path, 'cuda')
    # print('Policy reconstructed!')
    # episode_rewards, episode_lengths = \
    #     evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10000, deterministic=True, return_episode_rewards=True)
    # print('Evaluating policy')
    # episode_rewards, episode_lengths = \
    #     evaluate_policy(bc_policy, env, n_eval_episodes=100000, deterministic=False, return_episode_rewards=True, render=False)
    # print(f'Avg reward per episode: {np.mean(episode_rewards)}')
    # print(f'Avg episode length: {np.mean(episode_lengths)}')


if __name__ == '__main__':
    main()