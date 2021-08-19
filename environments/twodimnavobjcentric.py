import numpy as np
import gym
from gym import spaces

DIR = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

class TwoDimNavObjCentric(gym.Env):
	'''2D Navigation Environment Object Centric Graph Representation'''
	metadata = {'render.modes': ['human']}

	def __init__(self, size=10, num_objects=10, num_goals=1, goal_radius=1, goal_rew=10, max_time_step=100):
		# obj_set=[((0, 2), 0.5), ((1.2, 3.4), 0.2), ((5, 0, 1), ((2.3, 5.6), 0.6), ((4.1, 4.143), 0.139), ((5, 3.3), 0.26)], goal_set=[((8.2, 6.719), 1, 5), ((2, 8.3), 1, 12), ((7.2, 3.4), 0.5, 10)]):
		super(TwoDimNavObjCentric, self).__init__()
		self.size = size
		self.num_goals = num_goals
		self.goal_radius = goal_radius
		self.goal_rew = goal_rew
		self.max_time_step = max_time_step
		# self.obj_set = [np.random.uniform(low=-1, high=1, size=(2,)) for i in range(num_objects)]
		self.num_objects = num_objects
		self.num_nodes = 3

		self.pos = None
		self.goal_state = None
		self.goals_found = 0
		self.time_step = 0

		self.action_space = spaces.Discrete(len(DIR))
		self.num_node_features = 2

		self.reward_range = (0, goal_rew)
		self.full_conn_edge_idx = np.array([[], []])
		for i in range(num_objects):
			for j in range(num_objects):
				if i != j: 
					self.full_conn_edge_idx = np.concatenate((self.full_conn_edge_idx, np.array([[i], [j]])), axis=1)
		self.scene_edge_idx = np.array([[0, 1, 0, 2], [1, 0, 2, 0]])
		self.edge_idx = self.scene_edge_idx
		self.observation_dims = self.reset().shape
		self.observation_space = spaces.Box(low=-1, high=2, shape=self.observation_dims, dtype=np.float32)

	def _get_cur_state(self):
		node_state = []
		node_state = np.array([[0, 0], (self.pos/(self.size/2)) - 1, (self.goal_state/(self.size/2)) - 1])
		state = np.concatenate((node_state.flatten(), self.edge_idx.flatten()))
		return state

	def decode_state(self, state):
		node_state = state[:, 0, :self.num_objects]
		edge_index = state[:, 1:]
		return node_state, edge_index


	def reset(self):
		self.pos = np.array([int(self.size/2), int(self.size/2)])
		self.goal_state = np.random.uniform(low=0, high=self.size, size=(2,))
		self.goals_found = 0
		self.time_step = 0
		return self._get_cur_state()

	def step(self, action):
		action = DIR[action]
		self.pos = np.clip(self.pos + action, 0, self.size)
		reward = 0
		done = False
		self.time_step += 1
		if np.linalg.norm(self.pos - self.goal_state) < self.goal_radius: 
			reward += self.goal_rew
			self.goals_found += 1
			self.goal_state = np.random.uniform(low=0, high=self.size, size=(2,))
		done = (self.goals_found == self.num_goals) or (self.time_step >= self.max_time_step)
		return self._get_cur_state(), reward, done, {}

	def render(self, mode='human', close=False):
		print(f"Position: {self.pos}")
		print(f"Goal Position: {self.goal_state}, Radius: {self.goal_radius}, Reward: {self.goal_rew}")



if __name__ == '__main__':
	env = TwoDimNavObjCentric(max_time_step=10000000)
	time_steps = []
	for i in range(10):
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









