import numpy as np
import gym
from gym import spaces

# DIR = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 0])]

# DIR = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

DIR = [np.array([0, -1]), np.array([0, 1])]

class TwoDimNav(gym.Env):
	'''2D Navigation Environment'''
	metadata = {'render.modes': ['human']}

	def __init__(self, size=(10, 10), num_goals=2, goal_radius=1, goal_rew=10, max_time_step=100):
		super(TwoDimNav, self).__init__()
		self.size = np.array(size, dtype=int) # length and width of grid
		self.num_goals = num_goals 
		self.goal_radius = goal_radius # how big the goal is
		self.goal_rew = goal_rew # goal reward
		self.max_time_step = max_time_step
		# self.pos_discrete_goal_states = [np.array([0, 3]), np.array([3, 0]), np.array([2, 2]), np.array([1, 3]), np.array([4, 1])]
		# self.num_objects = len(self.pos_discrete_goal_states)
		self.num_node_features = 1
		# self.pos = np.random.uniform(low=0, high=self.size, size=(2,))
		self.pos = np.random.randint(size, size=(2,))
		# self.goal_state = np.random.uniform(low=0, high=size, size=(2,))
		self.goal_pos = np.random.randint(size, size=(2,))
		self.goal_state = np.concatenate((self.goal_pos, self.goal_pos))
		self.goals_found = 0
		self.time_step = 0

		self.action_space = spaces.Discrete(len(DIR))
		self.observation_size_high = np.concatenate((self.size, self.size))
		self.observation_space = spaces.Box(low=0, high=self.observation_size_high - 1,
			shape=(4,), dtype=np.float32)
		self.reward_range = (-1, goal_rew)

	def reset(self):
		# self.pos = np.array([int(self.size/2), int(self.size/2)])
		self.pos = np.random.randint(self.size, size=(2,))
		# self.goal_state = np.random.uniform(low=0, high=self.size, size=(2,))
		self.goal_pos = np.random.randint(self.size, size=(2,))
		self.goal_state = np.concatenate((self.goal_pos, self.goal_pos))
		self.goals_found = 0
		self.time_step = 0
		state = np.array([self.pos, self.goal_pos])
		state = state.flatten()
		return state

	def step(self, action):
		action = DIR[action]
		self.pos = np.clip(self.pos + action, 0, self.size - 1)
		reward = -1
		done = False
		self.time_step += 1
		if np.linalg.norm(self.pos - self.goal_pos) < self.goal_radius: 
			reward += self.goal_rew
			self.goals_found += 1
			# self.goal_state = np.random.uniform(low=0, high=self.size, size=(2,))
			# self.goal_pos = np.random.randint(self.size, size=(2,))
		done = (self.goals_found == self.num_goals) # or (self.time_step >= self.max_time_step)
		state = np.array([self.pos, self.goal_pos])
		state = state.flatten()
		return state, reward, done, {}

	def render(self, mode='human', close=False):
		print(f"Position: {self.pos}")
		print(f"Goal Position: {self.goal_pos}, Radius: {self.goal_radius}, Reward: {self.goal_rew}")



if __name__ == '__main__':
	env = TwoDimNav(max_time_step=10000000)
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









