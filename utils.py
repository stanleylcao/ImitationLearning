from environments.twodimnav import TwoDimNav
import pdb
import gym
import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
from heapq import heapify, heappush, heappop
from typing import Callable
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecEnv

def reconstruct_path(came_from, current) -> list:
    """
    Reconstructs the path from `current` using the `came_from` dict to
    retrace the steps 
    """
    total_path = []
    total_path.insert(0, current)
    while current in came_from: # check if `current` is in keys of `came_from`
        current = came_from[current]
        total_path.insert(0, current)
    return total_path

def a_star(env: TwoDimNav, start_pos: tuple, goal_pos: tuple, 
            h: Callable[[tuple, tuple], float]):
    """
    A* finds a path from `start_pos` to `goal_pos`. Both are tuples of
    shape (2,), which is ordered (current_x, current_y).
    `h` is a hueristic function. `h(n, goal)` estimates the cost to reach
    from node `n`.
    """
    # default value for the f_score and g_score dictionaries
    default_value = float('inf')
    
    # The set of discovered nodes that may need to be re-expanded.
    open_set = []
    heappush(open_set, (0, start_pos)) # Initially, only the start node is known.

    # For node `n`, `came_from[n]` is the node immediately preceding it on the
    # cheapest path from `start` to `n` currently known.
    came_from = {}

    # For node `n`, `g_score[n]` is the cost of the cheapest path from
    # `start` to `n` currently known
    g_score = {}
    g_score[start_pos] = 0

    # For node `n`, `f_score[n]` := `g_score[n] = h(n)`. `f_score[n]`
    # represents our current best guess as to how short a path from `start`
    # to `goal` can be if it goes through node `n`. 
    f_score = {}
    f_score[start_pos] = g_score[start_pos] + h(start_pos, goal_pos)

    while open_set: # empty list evaluated to False
        current_pos = heappop(open_set)[-1] # get last element instead of the priority
        if np.array_equal(current_pos, goal_pos): # check if current == goal
            env.current_pos = np.array(start_pos) # restore original position within environment
            return reconstruct_path(came_from, current_pos)
        for index, action in enumerate(env.actions):
            # Prevent a neighbor that will be out-of-bounds
            neighbor = tuple(np.clip(np.array(current_pos) + np.array(action),
                                0, env.grid_size - 1))
            if current_pos == neighbor:
                continue # disregard the action self.STAY
                
            env.current_pos = current_pos # set current position in env before stepping
            reward = env.get_reward_of_action(index)
            # TODO: implement a .step function that ignores negative
            # reward, since the only weights that matter are large positive
            # ones that possibly denote a wall or expensive obstruction.
            
            # The bigger the reward, the less weight (i.e., penalty) the
            # edge from `current` to `neighbor` actually has
            weight = -reward if reward < 0 else 1

            # tentative_g_score is the distance from `start` to the
            # `neighbor` of `current`
            tentative_g_score = g_score.get(current_pos, default_value) + weight
            if tentative_g_score < g_score.get(neighbor, default_value):
                # This path to neighbor is better than any previous one, so
                # we record it.
                came_from[neighbor] = current_pos
                g_score[neighbor] = tentative_g_score
                # import pdb; pdb.set_trace()
                if neighbor in map(lambda tup: tup[-1], open_set):
                    # f_score[neighbor] is still the old value at this
                    # point, so we remove the old neighbor with the old
                    # f_score before adding it back in with the new f_score
                    open_set.remove((f_score[neighbor], neighbor))
                    heapify(open_set)
                # Only now do we update the f_score
                f_score[neighbor] = g_score[neighbor] + \
                            h(neighbor, goal_pos)
                heappush(open_set, (f_score[neighbor], neighbor))

def linear_distance(start: tuple, end: tuple) -> float:
    return np.linalg.norm(np.array(end) - np.array(start))

def get_actions_from_path(env: TwoDimNav, path: list):
    """
    Recovers the action indices required to make the path. Returns this the
    action indices as a list.
    """
    action_indices = []
    if len(path) == 1:
        action_indices.append(env.get_action_index(env.STAY))
        return action_indices
    for i in range(len(path) - 1):
        action = np.array(path[i + 1]) - np.array(path[i])
        action_index = env.get_action_index(action)
        action_indices.append(action_index)
    return action_indices

def get_optimal_episode_data(env: TwoDimNav):
    """
    Uses the optimal policy (A* algorithm in the case of GridWorld), to step
    through `env`, returning the total episode reward and length
    """
    a_star_episode_reward = 0.0
    a_star_episode_length = 0
    for i in sorted(env.objectives):
        start_pos = tuple(env.current_pos)
        goal_pos = tuple(env.get_goal_pos_at_index(i))
        path = a_star(env, start_pos, goal_pos, linear_distance)
        action_indices = get_actions_from_path(env, path)
        for action_index in action_indices:
            obs, reward, done, _ = env.step(action_index)
            a_star_episode_reward += reward
            a_star_episode_length += 1
        env.current_pos = goal_pos

    assert done == True # Verify that the environment has been solved
    return a_star_episode_reward, a_star_episode_length

# TODO: see if copy_of_env is needed and update function documentation as needed
def evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param model: (BaseAlgorithm) The RL agent you want to evaluate.
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
    # num_models = len(models)
    # assert num_models > 0, 'You must pass in at least one model when using this function'
    
    # if num_models == 1:
    #     model = models[0]
    episode_rewards, episode_lengths = [], []
    a_star_episode_rewards, a_star_episode_lengths = [], []

    for _ in tqdm(range(n_eval_episodes)):
        obs = env.reset()
        assert isinstance(env, TwoDimNav)
        # Make copy of environment so that env_copy.step will not affect the env
        # that the model operates on
        env_copy = env.make_copy()
        a_star_episode_reward, a_star_episode_length = get_optimal_episode_data(env_copy)
        a_star_episode_rewards.append(a_star_episode_reward)
        a_star_episode_lengths.append(a_star_episode_length)

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
        if episode_length < a_star_episode_length:
            import pdb; pdb.set_trace()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    a_star_mean_reward = np.mean(a_star_episode_rewards)
    a_star_std_reward = np.std(a_star_episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return (episode_rewards, episode_lengths,
                a_star_episode_rewards, a_star_episode_lengths)

    return mean_reward, std_reward, a_star_mean_reward, a_star_std_reward

def create_evaluate_policy_callback(tb_writer: SummaryWriter, eval_env: gym.Env):
    def evaluate_policy_callback(local_var_dict: dict):
        """
        This is the callback passed into the `bc.train()` function to evaluate it
        until a certain average episode reward threshold.
        """
        model = local_var_dict['self'].policy
        env = eval_env
        print('Evaluating policy...')
        episode_rewards, episode_lengths, a_star_episode_rewards, a_star_episode_lengths= \
            evaluate_policy(model, env, n_eval_episodes=100 , deterministic=False, \
                                return_episode_rewards=True)
        
        # Record agent episode performance data to tensorboard
        avg_episode_reward = np.mean(episode_rewards)
        avg_episode_length = np.mean(episode_lengths)
        tb_writer.add_scalar('Agent/Average Episode Reward', avg_episode_reward, 
            global_step=local_var_dict['epoch_num'])
        tb_writer.add_scalar('Agent/Average Episode Length', avg_episode_length,
            global_step=local_var_dict['epoch_num'])

        # Record A* episode performance data to tensorboard
        a_star_avg_episode_reward = np.mean(a_star_episode_rewards)
        a_star_avg_episode_length = np.mean(a_star_episode_lengths)
        tb_writer.add_scalar('A* Algorithm/Average Episode Reward', a_star_avg_episode_reward, 
            global_step=local_var_dict['epoch_num'])
        tb_writer.add_scalar('A* Algorithm/Average Episode Length', a_star_avg_episode_length,
            global_step=local_var_dict['epoch_num'])

    return evaluate_policy_callback