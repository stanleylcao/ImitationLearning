import torch
import numpy as np

from imitation.algorithms import bc
from torch.utils.tensorboard import SummaryWriter
from tqdm.std import tqdm

from environments.twodimnav import TwoDimNav
from utils import evaluate_policy

logs_dir = 'logs/'

eval_logging_dir = logs_dir + 'BC/final_evaluation/'

path_suffix_onepolicy_approach = 'BC/onepolicy_approach/'
onepolicy_approach_save_path = 'policies/' + path_suffix_onepolicy_approach + 'policy'

path_suffix_subpolicy_approach = 'BC/subpolicy_approach/'
subpolicy_approach_save_dir = 'policies/' + path_suffix_subpolicy_approach

def create_final_eval_callback(tb_writer: SummaryWriter, approach_type: str):
    def eval_callback(locals: dict, globals: dict):
        """
        This callback is for the function `utils.evaluate_policy()`, mainly for
        tensorboard recording.
        """
        tb_writer.add_scalar(approach_type + '/Episode Reward', locals['episode_reward'],
            global_step=locals['episode_num'])
        tb_writer.add_scalar(approach_type + '/Episode Length', locals['episode_length'],
            global_step=locals['episode_num'])
    return eval_callback


def evaluate(n_eval_episodes=1000, size=(10, 10), num_goals=5, objectives=range(5)):
    env = TwoDimNav(size=size, num_goals=num_goals, objectives=objectives)
    eval_envs_onepolicy, eval_envs_subpolicy = [], []
    print(f'Generating {n_eval_episodes} evaluation environments...')
    for _ in tqdm(range(n_eval_episodes)):
        eval_envs_onepolicy.append(env.make_copy())
        eval_envs_subpolicy.append(env.make_copy())
        env.reset()

    print(f"All Tensorboards and logging are being written inside {eval_logging_dir}.")
    tb_writer = SummaryWriter(eval_logging_dir)

    # Get policy from one policy approach
    print('Reconstructing long horizon policy...')
    onepolicy = bc.reconstruct_policy(onepolicy_approach_save_path)
    print(f'Long horizon policy reconstructed on {onepolicy.device} device!')

    # Get all subpolicies from subpolicy approach
    subpolicies = []
    for goal_index in range(num_goals):
        print(f'Reconstructing subpolicy{goal_index}...')
        subpolicy_approach_save_path = subpolicy_approach_save_dir + f'subpolicy{goal_index}'
        subpolicy = bc.reconstruct_policy(subpolicy_approach_save_path)
        print(f'subpolicy{goal_index} reconstructed on {subpolicy.device} device!')
        subpolicies.append(subpolicy)
    
    # Evaluate one policy approach
    eval_callback = create_final_eval_callback(tb_writer, approach_type='One Policy Approach')
    mean_episode_reward, std_episode_reward, a_star_mean_episode_reward, a_star_std_episode_reward = \
        evaluate_policy(env, onepolicy, n_eval_episodes, eval_envs_onepolicy,
            deterministic=False, callback=eval_callback)
    print('-------- One Policy Approach --------')
    print(f'Mean Episode Reward: {mean_episode_reward}')
    print(f'Std Episode Reward: {std_episode_reward}')
    print(f'A* Mean Episode Reward: {a_star_mean_episode_reward}')
    print(f'A* Std Episode Reward: {a_star_std_episode_reward}')

    # Evaluate subpolicy approach
    eval_callback = create_final_eval_callback(tb_writer, 
        approach_type='Multi Subpolicy Approach')
    mean_episode_reward, std_episode_reward, a_star_mean_episode_reward, a_star_std_episode_reward = \
        evaluate_policy(env, subpolicies, n_eval_episodes, eval_envs_subpolicy,
            deterministic=False, callback=eval_callback)
    print('-------- Multi Subpolicy Approach --------')
    print(f'Mean Episode Reward: {mean_episode_reward}')
    print(f'Std Episode Reward: {std_episode_reward}')
    print(f'A* Mean Episode Reward: {a_star_mean_episode_reward}')
    print(f'A* Std Episode Reward: {a_star_std_episode_reward}')

if __name__ == '__main__':
    evaluate()