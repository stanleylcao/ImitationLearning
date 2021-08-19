import pathlib
from typing import Callable, Tuple
from torch.utils.data import Dataset
from heapq import heapify, heappush, heappop
from tqdm import tqdm
import tempfile

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from environments.twodimnav import TwoDimNav
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from imitation.algorithms import bc

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

RIGHT = np.array([0, 1])
LEFT = np.array([0, -1])
DOWN = np.array([1, 0])
UP = np.array([-1, 0])
STAY = np.array([0, 0])

DIR = [LEFT, RIGHT, DOWN, UP, STAY]

class ExpertDemos(Dataset):
    def __init__(self, size, num_paths) -> None:
        """
        Initializes the dataset with expert demos. `size` is the size of the
        gridworld. `num_paths` is the number of times to run the A* algorithm on
        grids. In other words, it is the number of gridworlds to spawn and
        solve.
        """
        np.random.seed(0) # Set seed for reproducibility
        self.env = TwoDimNav(size=size, num_goals=1)
        check_env(self.env) # validate environment
        self.heuristic = self.linear_distance
        self.obs, self.acts, self.next_obs, self.dones, self.infos = \
            self.generate_data(num_paths=num_paths)
    
    def __len__(self) -> int:
        return self.obs.shape[0]
    
    def __getitem__(self, index):
        return {'obs': self.obs[index],
                'acts': self.acts[index],
                'next_obs': self.next_obs[index],
                'dones': self.dones[index],
                'infos': self.infos[index]}
    
    def find_shortest_path(self) -> list:
        start_state_tuple = tuple(self.env.reset())
        goal_state_tuple = tuple(self.env.goal_state)
        # self.env.render()
        # pos_tuple = tuple(self.env.pos)
        # goal_tuple = tuple(self.env.goal_state)
        path = self.a_star(start_state_tuple, goal_state_tuple, self.linear_distance)
        return path

    @staticmethod
    def get_action_index(action: np.array):
        """
        Gets the index of the `action` within the `DIR` list.
        """
        return next(dir_index for dir_index in range(len(DIR)) 
                    if np.array_equal(action, DIR[dir_index]))
        

    def generate_data(self, num_paths=1) -> Tuple[torch.Tensor, torch.Tensor, 
        torch.Tensor, torch.Tensor, list]:
        observations, acts, next_obs, dones, infos = [], [], [], [], []
        print('Generating data...')
        for _ in tqdm(range(num_paths)):
            path = self.find_shortest_path()
            if (len(path) == 1):
                pass
                # observations.append(path[0])
                # acts.append(self.get_action_index(STAY))
                # next_obs.append(path[0])
                # dones.append(True)
                # infos.append({})
            else:
                for i in range(len(path) - 1):
                    observations.append(path[i])
                    action = np.array(path[i + 1][:2]) - np.array(path[i][:2])
                    acts.append(self.get_action_index(action))
                    next_obs.append(path[i + 1])
                    dones.append(path[i + 1][:2] == path[i + 1][2:])
                    infos.append({})
            self.env.reset()
        # TODO: convert to tensor last minute
        return (torch.tensor(observations),
                torch.tensor(acts),
                torch.tensor(next_obs),
                torch.tensor(dones),
                infos)

    @staticmethod
    def linear_distance(start: np.array, end: np.array) -> float:
        return np.linalg.norm(np.array(end) - np.array(start))

    def get_padded_action(self, action: np.array) -> np.array:
        """
        Since A* works on states rather than on the nodes themselves, the
        `action` must be right-extended with zeros
        """
        padded_action = np.zeros(self.env.observation_space.shape, dtype=int)
        padded_action[:action.shape[0]] = action
        return padded_action

    @staticmethod
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

    def a_star(self, start_state: tuple, goal_state: tuple, 
                h: Callable[[tuple, tuple], float]):
        """
        A* finds a path from `start_state` to `goal_state`. Both are tuples of
        shape (4,), which is ordered (current_x, current_y, goal_x, goal_y).
        `h` is a hueristic function. `h(n, goal)` estimates the cost to reach
        from node `n`.
        """
        # default value for the f_score and g_score dictionaries
        default_value = float('inf')
        
        # The set of discovered nodes that may need to be re-expanded.
        open_set = []
        heappush(open_set, (0, start_state)) # Initially, only the start node is known.

        # For node `n`, `came_from[n]` is the node immediately preceding it on the
        # cheapest path from `start` to `n` currently known.
        came_from = {}

        # For node `n`, `g_score[n]` is the cost of the cheapest path from
        # `start` to `n` currently known
        g_score = {}
        g_score[start_state] = 0

        # For node `n`, `f_score[n]` := `g_score[n] = h(n)`. `f_score[n]`
        # represents our current best guess as to how short a path from `start`
        # to `goal` can be if it goes through node `n`. 
        f_score = {}
        f_score[start_state] = g_score[start_state] + h(start_state[2:], goal_state[2:])

        while open_set: # empty list is false
            current_state = heappop(open_set)[-1] # get last element instead of the priority
            if np.array_equal(current_state, goal_state): # check if current == goal
                return self.reconstruct_path(came_from, current_state)
            for index, action in enumerate(DIR):
                padded_action = self.get_padded_action(action)
                # Prevent a neighbor that will be out-of-bounds
                neighbor_state = tuple(np.clip(np.array(current_state) + np.array(padded_action),
                                    0, self.env.observation_size_high - 1))
                if current_state == neighbor_state:
                    continue
                
                self.env.pos = current_state[:2] # set current position in env before stepping
                state, reward, dones, infos = self.env.step(index)
                # The bigger the reward, the less weight (i.e., penalty) the
                # edge from `current` to `neighbor` actually has
                weight = -reward

                # tentative_g_score is the distance from `start` to the
                # `neighbor` of `current`
                tentative_g_score = g_score.get(current_state, default_value) + weight
                if tentative_g_score < g_score.get(neighbor_state, default_value):
                    # This path to neighbor is better than any previous one, so
                    # we record it.
                    came_from[neighbor_state] = current_state
                    g_score[neighbor_state] = tentative_g_score
                    f_score[neighbor_state] = g_score[neighbor_state] + \
                                                    h(neighbor_state[:2], goal_state[:2])
                    # print(neighbor_state, open_set) # TODO remove
                    if neighbor_state in map(lambda tup: tup[-1], open_set):
                        open_set.remove(neighbor_state)
                        heapify(open_set)
                    heappush(open_set, (f_score[neighbor_state], neighbor_state))

def main():
    expert_demos = ExpertDemos(size=(1, 10), num_paths=10000)
    data_loader = DataLoader(expert_demos, batch_size=32)
    env = expert_demos.env
    
    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")
    import pdb; pdb.set_trace()
    
    # Train BC on expert data.
    # BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
    # dictionaries containing observations and actions.
    logger.configure(tempdir_path / "BC/")

    bc_trainer = bc.BC(env.observation_space, env.action_space, expert_data=data_loader)
    bc_trainer.train(n_epochs=15, log_interval=1)
    print(f'Num of expert demos: {len(data_loader.dataset)}')
    episode_rewards, episode_lengths = \
        evaluate_policy(bc_trainer.policy, env, n_eval_episodes=20, deterministic=True, return_episode_rewards=True)
    print(f'Avg reward per episode: {np.mean(episode_rewards)}')
    print(f'Avg episode length: {np.mean(episode_lengths)}')

    # obs, acts, next_obs, dones, infos = expert_demos.get_relevant_tensors(num_paths=2)
    # print(obs)
    # print(acts)
    # print(next_obs)
    # print(dones)
    # print(infos)

if __name__ == '__main__':
    main()