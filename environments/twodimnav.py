from heapq import heappush
from typing import Iterable, Union
import numpy as np
import gym
from gym import spaces
from numpy.linalg import norm


# DIR = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

# DIR = [np.array([0, -1]), np.array([0, 1])]

# TODO: Rename to GridWorld
class TwoDimNav(gym.Env):
    '''2D Navigation Environment'''
    metadata = {'render.modes': ['human']}

    def __init__(self, size=(10, 10), num_goals=3, objectives: Iterable[int]=None, 
        goal_radius=1, goal_rew=10, max_time_step=None):
        """
        `objectives` is a list of indices referring to the goals that need to be
        reached in order to have solved the environment
        """
        super(TwoDimNav, self).__init__()
        # Define actions
        self.LEFT = np.array([0, -1])
        self.RIGHT = np.array([0, 1])
        self.UP = np.array([-1, 0])
        self.DOWN = np.array([1, 0])
        self.STAY = np.array([0, 0])
        self.actions = [self.LEFT, self.RIGHT, self.UP, self.DOWN, self.STAY]

        self.grid_size = np.array(size, dtype=int) # length and width of grid
        self.num_goals = num_goals 
        # ensure enough room for distinct goals
        assert self.num_goals <= np.prod(self.grid_size)
        if objectives is None:
            self.objectives = list(range(num_goals))
        else:
            self.objectives = list(objectives)
        
        self.goal_radius = goal_radius # how big the goal is
        self.goal_rew = goal_rew # goal reward
        self.goals_found = 0
        self.goals_reached = []
        self.time_step = 0
        if max_time_step is None:
            self.max_time_step = np.prod(self.grid_size) * 2
        else:
            self.max_time_step = max_time_step
        # goals have 3 features: the x coord, the y coord, and whether the goal
        # is completed (denoted by 0 or 1 for uncompleted and completed,
        # respectively)
        self.goal_features_max = np.append(self.grid_size, 2)
        self.goals = self.create_goals()

        # the max value of the any observation. This is used to specify the
        # observation space. It contains the current position and all 3-tuples
        # denoting the features of each goal.
        self.observation_size_high = self.grid_size
        for _ in range(self.num_goals):
            self.observation_size_high = np.append(self.observation_size_high, 
                                            self.goal_features_max)

        self.current_pos = np.random.randint(self.grid_size, size=self.grid_size.shape)
        self.action_space = spaces.Discrete(len(self.actions))
        # (10, 10, 10, 10, 2, 10, 10, 2, ...) TODO: remove
        self.observation_space = spaces.Box(low=0, high=self.observation_size_high - 1,
            shape=self.observation_size_high.shape, dtype=np.float32)

    def get_action_index(self, action: np.array):
        """
        Gets the index of the `action` within the `actions` list.
        """
        return next(dir_index for dir_index in range(len(self.actions)) 
                    if np.array_equal(action, self.actions[dir_index]))

    def make_copy(self) -> 'TwoDimNav':
        """
        Returns a copy of the instance `self`, having the same current position
        and goal position(s).
        """
        # Check that constructor parameters are indeed the same
        copied_env = TwoDimNav(self.grid_size, self.num_goals, self.objectives,
                    self.goal_radius, self.goal_rew, self.max_time_step)
        # Copy current position and goal position(s)
        copied_env.current_pos = np.array(self.current_pos)
        copied_env.goals = np.array(self.goals)
        return copied_env
    
    def is_repeated_goal(self, generated_goal_positions: np.array, 
        potential_goal_pos: np.array):
        """
        For n generated goals, the parameter `generated_goal_positions` is a
        (n, self.grid_size.shape[0]) numpy array containing all previously
        generated goal positions. The parameter `potential_goal_pos` is the new
        goal position to potentially add. This function returns True iff
        `potential_goal_pos` is not in one of the rows of
        `generated_goal_positions`.
        """
        # Create array of booleans; each element is True iff the corresponding
        # position coordinate matches (via broadcasting)
        matching_elems = (generated_goal_positions == potential_goal_pos)
        
        # A full match would have an entire row full of True
        row_sums = np.sum(matching_elems, axis=1)
        full_matches = (row_sums == self.grid_size.shape[0]) # grid_size.shape = (2,) for 2D
        num_full_matches = np.sum(full_matches)
        return num_full_matches == 0 # True iff there are no full matches
        
    def create_goals(self):
        """
        Creates `self.num_goals` number of distinct goals.
        """
        goals = np.empty(shape=(0, self.goal_features_max.shape[0]), dtype=int)
        for _ in range(self.num_goals):
            goal = np.random.randint(self.grid_size, size=self.grid_size.shape)
            while not self.is_repeated_goal(goals[:, :-1], goal):
                goal = np.random.randint(self.grid_size, size=self.grid_size.shape)
            goal = np.append(goal, 0).reshape(1, -1) # set goal as uncompleted by default
            goals = np.append(goals, goal, axis=0)
        return goals
    
    def mark_goals_incomplete(self, goal_indices: Iterable):
        """
        Marks the goals within the iterable `goal_indices` as "incomplete", which means
        that the third feature in the goal 3-tuple is set to 0. The iterable
        `goal_indices` should contain the indices of the goals to reset, which is
        0-based.
        """
        self.time_step = 0
        completeness_index = self.goals.shape[1] - 1 # TODO: change to -1
        for i in range(len(goal_indices)):
            goal_index = goal_indices[i]
            # Get the index of the bit representing whether the goal is
            # completed 
            self.goals[goal_index, completeness_index] = 0

    def is_goal_at_index_complete(self, index: int):
        return self.goals[index, -1]

    def get_goal_pos_at_index(self, index: int):
        # goal position is the first two numbers/features of the 3-tuple at
        # index `index`
        return self.goals[index, :-1]

    def get_reward_of_action(self, action_index: int):
        """
        This function gets the reward that the agent would earn if it takes the
        action referred to by the action_index. This function does not change
        any internal variables, nor does it update the state.
        """
        action = self.actions[action_index]
        new_pos = np.clip(self.current_pos + action, 0, self.grid_size - 1)
        goals_currently_at = (np.linalg.norm(new_pos - self.goals[:, :-1], axis=1)
                            < self.goal_radius)
        reward = -1
        for i in range(self.num_goals):
            if goals_currently_at[i] and self.goals[i, -1] == 0:
                reward = self.goal_rew
        return reward

    def reset(self):
        self.current_pos = np.random.randint(self.grid_size, size=self.grid_size.shape)
        self.goals = self.create_goals()
        self.goals_found = 0
        self.time_step = 0
        self.goals_reached.clear()
        state = np.append(self.current_pos, self.goals)
        return state

    def step(self, action_index):
        action = self.actions[action_index]
        self.current_pos = np.clip(self.current_pos + action, 0, self.grid_size - 1)
        reward = -1
        self.time_step += 1
        done = False
        # np.array of boolean values that are True iff the goal at that index
        # has been reached
        goals_currently_at = (np.linalg.norm(self.current_pos - self.goals[:, :-1], axis=1)
                            < self.goal_radius)
        
        for i in self.objectives: # multiple goals could be reached with a larger radius
            # Following logic only counts the reward and completes the goal if
            # the goal was completed in the order seen in `self.objectives`.
            if goals_currently_at[i] and self.goals[i, -1] == 0:
                self.goals[i, -1] = 1 # set goal as completed
                self.goals_found += 1 
                self.goals_reached.append(i)
                reward = self.goal_rew
            elif i not in self.goals_reached:
                break

        done = bool(self.objectives == self.goals_reached) \
            or (self.time_step >= self.max_time_step)
        state = np.append(self.current_pos, self.goals)
        return state, reward, done, {}

    def render(self, mode='human', close=False):
        print(f'Current position:\t{self.current_pos}')
        for i in range(self.num_goals):
            print(f'Goal {i} position:\t{self.goals[i, :-1]}\t Completed: {self.goals[i, -1]}')
        # print(f"Goal Position: {self.goals}, Radius: {self.goal_radius}, Reward: {self.goal_rew}")



if __name__ == '__main__':
    env = TwoDimNav()
    import pdb; pdb.set_trace()
    time_steps = []
    for i in range(1):
        print("---------------------------------------------\n\n")
        done = False
        j = 0
        env.reset()
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            j += 1
            print(f"Episode: {i}, Time Step: {j}, Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
            if j > 100000:
                break
        time_steps.append(j)
    print(f"Average time: {sum(time_steps)/len(time_steps)}")









