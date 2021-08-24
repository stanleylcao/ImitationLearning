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

# TODO: see if copy_of_env is needed and update function documentation as needed
def evaluate_policy(
    model,
    env,
    copy_of_env=None,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
    tensorboard=None
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseAlgorithm) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param copy_of_env: (gym.Env) The environment that an optimal policy will
        use to compare episodic reward. It is another `gym.Env` instantiated 
            with the same constructor parameters as `env`
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :param tensorboard: (torch.utils.tensorboard.SummaryWriter) the tensorboard
        to update after the policy has been evaluated.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    from tqdm import tqdm # TODO move above 
    for _ in tqdm(range(n_eval_episodes)):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths

    return mean_reward, std_reward

def create_evaluate_policy_callback(tb_writer: SummaryWriter, eval_env: gym.Env):
    def evaluate_policy_callback(local_var_dict: dict):
        """
        This is the callback passed into the `bc.train()` function to evaluate it
        until a certain average episode reward threshold.
        """
        model = local_var_dict['self'].policy
        env = eval_env
        episode_rewards, episode_lengths = \
            evaluate_policy(model, env, n_eval_episodes=500, deterministic=False, \
                                return_episode_rewards=True)
        avg_episode_reward = np.mean(episode_rewards)
        tb_writer.add_scalar('Average Episode Reward', avg_episode_reward, 
            global_step=local_var_dict['epoch_num'])
    return evaluate_policy_callback

def create_expert_demos(generate_new_data: bool, expert_demos_filename: str,
    grid_size=(10, 10), num_paths=5000, num_goals=1, start_goal_index=0, end_goal_index=0):
    """
    Creates expert demos based on whether new data is requested and whether the
    pickled file for the expert demos exists. See constructor of ExpertDemos for
    more information about the parameters.
    """
    if not generate_new_data and os.path.isfile(expert_demos_filename):
        print('No request for new data and data file exists!\nLoading data...')
        with open(expert_demos_filename, 'rb') as f:
            expert_demos = pickle.load(f)
    else:
        expert_demos = ExpertDemos(size=grid_size, num_paths=num_paths, num_goals=num_goals, 
            goal_to_start_from=start_goal_index, goal_to_end_at=end_goal_index)
        with open(expert_demos_filename, 'wb') as f:
            pickle.dump(expert_demos, f)
    return expert_demos

def main():
    np.random.seed(0)
    grid_size = (10, 10)
    num_paths = 5000
    num_goals = 3
    start_goal_index = 0
    end_goal_index = 2
    expert_demos_filename = \
        f'data/expert_demos({grid_size},{num_paths},{num_goals},{start_goal_index},{end_goal_index}).pkl'
    
    # change if you want to guarantee the generation of new data
    generate_new_data = True
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
    #     print(obs[i], '\t', next_obs[i])
    # import pdb; pdb.set_trace()
    data_loader = DataLoader(expert_demos, batch_size=32, shuffle=True)
    env = expert_demos.env

    logs_dir = 'logs/'
    path_suffix = 'BC/onepolicy_approach'
    
    BC_logging_path = logs_dir + path_suffix
    print(f"All Tensorboards and logging are being written inside {BC_logging_path}/.")
    
    policy_save_path = 'policies/' + path_suffix
    print(f'Policies to be saved in {policy_save_path}/')
    # import pdb; pdb.set_trace()

    # Configure logger to write with tensorboard and to stdout 
    logger.configure(folder=BC_logging_path, format_strs=['tensorboard', 'stdout'])
    
    # Train BC on expert data.
    bc_trainer = bc.BC(env.observation_space, env.action_space, expert_data=data_loader)
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
    bc_policy = bc.reconstruct_policy(policy_save_path, 'cuda')
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