import pickle, os.path
from typing import Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import numpy as np
from environments.twodimnav import TwoDimNav
from stable_baselines3.common.env_checker import check_env

from utils import a_star, linear_distance

class ExpertDemos(Dataset):
    def __init__(self, size, num_paths, num_goals, 
        goal_to_start_from=0, goal_to_end_at=0):
        """
        Initializes the dataset with expert demos. `size` is the size of the
        gridworld. `num_paths` is the number of times to run the A* algorithm on
        grids; in other words, it is the number of gridworlds to spawn and
        solve. `num_goals` is the number of goals in the GridWorld environment
        `goal_to_start_from` is the index of the first goal to reach, which is 0
        based. `goal_to_end_at` is the index of the last goal to reach, which is
        included in the sequential progression
        """
        assert 0 <= goal_to_start_from <= goal_to_end_at < num_goals
        np.random.seed(1) # Set seed for reproducibility
        import random; random.seed(0)
        self.size = size
        self.num_paths = num_paths
        self.num_goals_in_env = num_goals
        self.start_goal_index = goal_to_start_from
        self.end_goal_index = goal_to_end_at
        
        self.env = TwoDimNav(size=self.size, num_goals=self.num_goals_in_env, 
            objectives=range(goal_to_start_from, goal_to_end_at + 1))
        check_env(self.env) # validate environment

        self.heuristic = linear_distance # TODO: change to manhattan distance
        self.obs, self.acts, self.next_obs, self.dones, self.infos = \
            self.generate_data()
    
    def __len__(self) -> int:
        return self.obs.shape[0]
    
    def __getitem__(self, index):
        return {'obs': self.obs[index],
                'acts': self.acts[index],
                'next_obs': self.next_obs[index],
                'dones': self.dones[index],
                'infos': self.infos[index]}
        
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, 
        torch.Tensor, torch.Tensor, list]:
        observations, acts, next_obs, dones, infos = [], [], [], [], []
        print('Generating data...')
        for _ in tqdm(range(self.num_paths)):
            for goal_index in range(self.start_goal_index, self.end_goal_index + 1):
                goal_pos = self.env.get_goal_pos_at_index(goal_index)
                if goal_index == self.start_goal_index:
                    start_pos = self.env.current_pos
                else: # always start at previous goal
                    start_pos = self.env.get_goal_pos_at_index(goal_index - 1)
                path = a_star(self.env, tuple(start_pos), tuple(goal_pos), self.heuristic)
                
                # set the current position to the start of the path
                # self.env.current_pos = path[0]
                # Always need to append the initial state
                state = np.append(self.env.current_pos, self.env.goals)
                if len(path) == 1:
                    action_index = self.env.get_action_index(self.env.STAY)
                    next_state, _, done, _ = self.env.step(action_index)
                    observations.append(state)
                    acts.append(action_index)
                    next_obs.append(next_state)
                    dones.append(done)
                    infos.append({})
                else:
                    for i in range(len(path) - 1):
                        action = np.array(path[i + 1]) - np.array(path[i])
                        action_index = self.env.get_action_index(action)
                        next_state, _, done, _ = self.env.step(action_index)
                        observations.append(state)
                        acts.append(action_index)
                        next_obs.append(next_state)
                        dones.append(done)
                        infos.append({})
                        state = next_state

                    # for i in range(len(path) - 1):
                    #     observations.append(path[i])
                    #     action = np.array(path[i + 1][:2]) - np.array(path[i][:2])
                    #     acts.append(self.get_action_index(action))
                    #     next_obs.append(path[i + 1])
                    #     dones.append(path[i + 1][:2] == path[i + 1][2:])
                    #     infos.append({})
            self.env.reset()
        # TODO: convert to tensor last minute
        return (torch.tensor(observations),
                torch.tensor(acts),
                torch.tensor(next_obs),
                torch.tensor(dones),
                infos)

    

    # def get_padded_action(self, action: np.array) -> np.array:
    #     """
    #     Since A* works on states rather than on the nodes themselves, the
    #     `action` must be right-extended with zeros
    #     """
    #     padded_action = np.zeros(self.env.observation_space.shape, dtype=int)
    #     padded_action[:action.shape[0]] = action
    #     return padded_action

def create_expert_demos(generate_new_data: bool, expert_demos_filename: str,
    grid_size=(10, 10), num_paths=5000, num_goals=1, start_goal_index=0, end_goal_index=0):
    """
    Creates expert demos based on whether new data is requested and whether the
    pickled file for the expert demos exists. See constructor of ExpertDemos for
    more information about the parameters.
    """
    if not generate_new_data and os.path.isfile(expert_demos_filename):
        print('No request for new data and data file exists!')
        print(f'Loading data from {expert_demos_filename} ...')
        with open(expert_demos_filename, 'rb') as f:
            expert_demos = pickle.load(f)
    else:
        expert_demos = ExpertDemos(size=grid_size, num_paths=num_paths, num_goals=num_goals, 
            goal_to_start_from=start_goal_index, goal_to_end_at=end_goal_index)
        with open(expert_demos_filename, 'wb') as f:
            pickle.dump(expert_demos, f)
    return expert_demos