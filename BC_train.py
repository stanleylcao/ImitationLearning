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

# TODO: remove
RIGHT = np.array([0, 1])
LEFT = np.array([0, -1])
DOWN = np.array([1, 0])
UP = np.array([-1, 0])
STAY = np.array([0, 0])

# DIR = [LEFT, RIGHT, DOWN, UP, STAY]


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
        self.num_paths = num_paths
        self.num_goals_in_env = num_goals
        self.start_goal_index = goal_to_start_from
        self.end_goal_index = goal_to_end_at
        
        self.env = TwoDimNav(size=size, num_goals=self.num_goals_in_env)
        check_env(self.env) # validate environment

        self.heuristic = self.linear_distance # TODO: change to manhattan distance
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
    
    # TODO: remove function?
    def find_shortest_path(self, goal_index: int) -> list:
        """
        Feeds the correct starting position and goal position to the A*
        algorithm. `goal_index` is the index of the goal to find if there are
        multiple goals.
        """
        # TODO: remove
        # self.env.render()
        # pos_tuple = tuple(self.env.pos)
        # goal_tuple = tuple(self.env.goal_state)
        start_pos = self.env.current_pos
        goal_pos = self.env.get_goal_pos_at_index(self.start_goal_index)

        path = self.a_star(start_pos, goal_pos, self.heuristic)
        return path

    def get_action_index(self, action: np.array):
        """
        Gets the index of the `action` within the `DIR` list.
        """
        return next(dir_index for dir_index in range(len(self.env.actions)) 
                    if np.array_equal(action, self.env.actions[dir_index]))
        

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
                path = self.a_star(tuple(start_pos), tuple(goal_pos), self.heuristic)
                
                # Set actually incomplete goals to be incomplete in order to
                # correctly initialize the environment that was possible changed
                # by the `a_star` function via the calls to `.step`
                self.env.mark_goals_incomplete(range(goal_index, self.num_goals_in_env))
                # set the current position to the start of the path
                self.env.current_pos = path[0]
                # Always need to append the initial state
                state = np.append(self.env.current_pos, self.env.goals)
                if len(path) == 1:
                    action_index = self.get_action_index(STAY)
                    next_state, _, done, _ = self.env.step(action_index)
                    observations.append(state)
                    acts.append(action_index)
                    next_obs.append(state)
                    dones.append(done)
                    infos.append({})
                else:
                    for i in range(len(path) - 1):
                        action = np.array(path[i + 1]) - np.array(path[i])
                        action_index = self.get_action_index(action)
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

    @staticmethod
    def linear_distance(start: np.array, end: np.array) -> float:
        return np.linalg.norm(np.array(end) - np.array(start))

    # def get_padded_action(self, action: np.array) -> np.array:
    #     """
    #     Since A* works on states rather than on the nodes themselves, the
    #     `action` must be right-extended with zeros
    #     """
    #     padded_action = np.zeros(self.env.observation_space.shape, dtype=int)
    #     padded_action[:action.shape[0]] = action
    #     return padded_action

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

    def a_star(self, start_pos: tuple, goal_pos: tuple, 
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
            # import pdb; pdb.set_trace()
            current_pos = heappop(open_set)[-1] # get last element instead of the priority
            if np.array_equal(current_pos, goal_pos): # check if current == goal
                return self.reconstruct_path(came_from, current_pos)
            for index, action in enumerate(self.env.actions):
                # Prevent a neighbor that will be out-of-bounds
                neighbor = tuple(np.clip(np.array(current_pos) + np.array(action),
                                    0, self.env.grid_size - 1))
                if current_pos == neighbor:
                    continue # disregard the action self.STAY
                
                self.env.current_pos = current_pos # set current position in env before stepping
                state, reward, done, infos = self.env.step(index)

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

def main():
    np.random.seed(0)
    expert_demos = ExpertDemos(size=(10, 10), num_paths=5000, num_goals=3, goal_to_end_at=2)
    # obs = expert_demos.obs
    # acts = expert_demos.acts
    # next_obs = expert_demos.next_obs
    # dones = expert_demos.dones

    # for i in range(obs.shape[0]):
    #     print(obs[i], '\t', next_obs[i])
    # import pdb; pdb.set_trace()
    data_loader = DataLoader(expert_demos, batch_size=32)
    env = expert_demos.env
    
    # tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    # tempdir_path = pathlib.Path(tempdir.name)
    # print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")
    # # import pdb; pdb.set_trace()
    
    # # Train BC on expert data.
    # # BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
    # # dictionaries containing observations and actions.
    # logger.configure(tempdir_path / "BC/")

    bc_trainer = bc.BC(env.observation_space, env.action_space, expert_data=data_loader)
    bc_trainer.train(n_epochs=80, log_interval=1)
    print(f'Num of expert demos: {len(data_loader.dataset)}')
    episode_rewards, episode_lengths = \
        evaluate_policy(bc_trainer.policy, env, n_eval_episodes=100, deterministic=True, return_episode_rewards=True)
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