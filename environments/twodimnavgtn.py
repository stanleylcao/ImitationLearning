import numpy as np
import gym
from gym import spaces

DIR = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

class TwoDimNavGTN(gym.Env):
	'''2D Navigation Environment Object Centric Graph Representation'''
	metadata = {'render.modes': ['human']}

	def __init__(self, num_goals=2, object_num=[((1, 2), [10]), ((3, 1), [1, 5, 12]), ((0, 5), [])], determined_start_pos=True, determined_obj=True, size=10, goal_radius=0.8, goal_rew=10, max_time_step=50):
		# obj_set=[((0, 2), 0.5), ((1.2, 3.4), 0.2), ((5, 0, 1), ((2.3, 5.6), 0.6), ((4.1, 4.143), 0.139), ((5, 3.3), 0.26)], goal_set=[((8.2, 6.719), 1, 5), ((2, 8.3), 1, 12), ((7.2, 3.4), 0.5, 10)]):
		super(TwoDimNavGTN, self).__init__()
		self.size = size
		self.goal_radius = goal_radius
		self.num_goals = num_goals
		self.goal_rew = goal_rew
		self.max_time_step = max_time_step
		self.determined_start_pos = determined_start_pos
		self.determined_obj = determined_obj

		# self.obj_set = [np.random.uniform(low=-1, high=1, size=(2,)) for i in range(num_objects)]
		# self.blue_obj = np.array([[0.9 * size, 0.2 * size]])
		# self.blue_goal = np.array([[0.1 * size, 0.1 * size], [0.1 * size, 0.6 * size]])
		# self.blue_obj = np.concatenate((self.blue_goal, self.blue_obj), axis=0)
		# self.red_obj = np.array([[0.2 * size, 0.3 * size], [0.6 * size, 0.4 * size], [0.9 * size, 0.8 * size]])
		# self.red_goal = np.array([[0.3 * size, 0.4 * size]])
		# self.red_obj = np.concatenate((self.red_goal, self.red_obj), axis=0)
		# self.distract_obj = np.array([[0 * size, 0 * size], [0.8 * size, 0.8 * size], [0.2 * size, 0.5 * size], [0.3 * size, 0.7 * size], [0.8 * size, 0.1 * size]])
		# self.obj_set = np.concatenate((self.blue_obj, self.red_obj, self.distract_obj), axis=0)

		self.object_num = object_num
		self.num_class = len(object_num)
		self.n_goals = []
		self.n_distracts = []
		self.n_objs = []
		# self.obj_set = np.array([[]])
		self.obj_set = np.empty(shape=(0, 2))
		# self.goal_set = np.array([[]])
		self.goal_set = np.empty(shape=(0, 2))
		self.goal_rewards = []
		for n_obj, goal_rew in object_num: 
			n_goal, n_distract = n_obj
			goal_obj = np.random.uniform(low=0, high=size, size=(n_goal, 2))
			distract_obj = np.random.uniform(low=0, high=size, size=(n_distract, 2))
			obj = np.concatenate((goal_obj, distract_obj), axis=0)
			self.n_goals.append(n_goal)
			self.n_distracts.append(n_distract)
			self.n_objs.append(n_goal + n_distract)
			self.goal_rewards.extend(goal_rew)
			# print(self.goal_set, goal_obj)
			# print(self.goal_set.shape, goal_obj.shape)
			self.goal_set = np.concatenate((self.goal_set, goal_obj), axis=0)
			self.obj_set = np.concatenate((self.obj_set, obj), axis=0)

		self.pos = [5, 5]
		# self.goal_states = np.concatenate((self.blue_goal, self.red_goal), axis=0) 
		self.goals_found = 0
		self.time_step = 0

		self.num_objects = self.obj_set.shape[0]
		self.num_goal_objects = self.goal_set.shape[0]
		assert(len(self.goal_rewards) == self.num_goal_objects)
		# self.num_blue_objects = self.blue_obj.shape[0]
		# self.num_blue_goals = self.blue_goal.shape[0]
		# self.num_red_objects = self.red_obj.shape[0]
		# self.num_red_goals = self.red_goal.shape[0]
		# self.num_distract_objects = self.distract_obj.shape[0]
		# self.num_goal_objects = self.goal_states.shape[0]
		self.num_nodes = self.num_objects + 2

		self.action_space = spaces.Discrete(len(DIR))

		self.reward_range = (0, goal_rew)
		self.scene_edge_idx = np.array([[], []])
		# print(self.num_objects + 1)
		for i in range(1, self.num_nodes):
			self.scene_edge_idx = np.concatenate((self.scene_edge_idx, np.array([[0], [i]])), axis=1)
			self.scene_edge_idx = np.concatenate((self.scene_edge_idx, np.array([[i], [0]])), axis=1)
		self.edge_idx = self.scene_edge_idx

		self.adj_matrices = []
		adj_mat = np.zeros((self.num_nodes, self.num_nodes))
		adj_mat[0][1] = 1
		adj_mat[1][0] = 1
		self.adj_matrices.append(adj_mat)
		idx = 2
		for obj_i, obj_num in enumerate(self.n_objs):
			adj_mat = np.zeros((self.num_nodes, self.num_nodes))
			for i in range(obj_num):
				adj_mat[0][idx] = 1
				adj_mat[idx][0] = 1
				idx += 1
			self.adj_matrices.append(adj_mat)
		# print("The adjacency matrices shape is ", self.adj_matrices.shape)

		# print(self.reset(), self.edge_idx)
		# exit(-1)
		self.num_node_features = self.reset().size//self.num_nodes
		self.observation_dims = self.reset().shape
		print("\n\nThe observation dimensions are ", self.observation_dims, "\n\n")
		self.observation_space = spaces.Box(low=-1, high=3, shape=self.observation_dims, dtype=np.float32)

	def _get_cur_state(self):
		node_state = np.array([0, 0, 0, 0])
		# print(list((self.pos/(self.size/2)) - 1))
		# print(np.concatenate(((self.pos/(self.size/2)) - 1, [0, 0]), axis=0))
		# print(list((self.pos/(self.size/2)) - 1).extend([0, 0]))
		node_state = np.concatenate((node_state, np.concatenate(((self.pos/(self.size/2)) - 1, [0, 0]), axis=0)))
		idx = 0
		for obj_i, obj_num in enumerate(self.n_objs): 
			goal_num = self.n_goals[obj_i]
			for i in range(obj_num):
				is_goal = int(i < goal_num)
				node_state = np.concatenate((node_state, np.concatenate(((self.obj_set[idx]/(self.size/2)) - 1, [obj_i + 1, is_goal]), axis=0)))
				idx += 1

		# for i in range(self.num_blue_objects):
		# 	is_goal = int(i < self.num_blue_goals)
		# 	node_state = np.concatenate((node_state, np.concatenate(((self.blue_obj[i]/(self.size/2)) - 1, [1, is_goal]), axis=0)))
		# 	# node_state = np.concatenate((node_state, list(self.blue_obj[i]/(self.size/2) - 1).extend[1, is_goal]))
		# for i in range(self.num_red_objects):
		# 	is_goal = int(i < self.num_red_goals)
		# 	node_state = np.concatenate((node_state, np.concatenate(((self.red_obj[i]/(self.size/2)) - 1, [2, is_goal]), axis=0)))
		# 	# node_state = np.concatenate((node_state, list(self.red_obj[i]/(self.size/2) - 1).extend[2, is_goal]))
		# for i in range(self.num_distract_objects):
		# 	node_state = np.concatenate((node_state, np.concatenate(((self.distract_obj[i]/(self.size/2)) - 1, [3, 0]), axis=0)))
		# 	# node_state = np.concatenate((node_state, list(self.distract_obj[i]/(self.size/2) - 1).extend[3, 0]))
		# print(node_state.shape)
		# state = np.concatenate((node_state.flatten(), self.edge_idx.flatten()))
		state = node_state.flatten()
		return state

	def decode_state(self, state):
		node_state = state[:, 0, :self.num_objects]
		edge_index = state[:, 1:]
		return node_state, edge_index


	def reset(self):
		# self.pos = np.array([int(self.size/2), int(self.size/2)])
		# self.goal_state = np.random.uniform(low=0, high=self.size, size=(2,))
		# self.pos = np.array([5, 5])
		if not self.determined_obj:
			# self.obj_set = np.array([[]])
			# self.goal_set = np.array([[]])
			self.obj_set = np.empty(shape=(0, 2))
			self.goal_set = np.empty(shape=(0, 2))

			for n_goal, n_distract in self.object_num: 
				goal_obj = np.random.uniform(low=0, high=self.size, size=(n_goal, 2))
				distract_obj = np.random.uniform(low=0, high=self.size, size=(n_distract, 2))
				obj = np.concatenate((goal_obj, distract_obj), axis=0)
				self.goal_set = np.concatenate((self.goal_set, goal_obj), axis=0)
				self.obj_set = np.concatenate((self.obj_set, obj), axis=0)
		self.pos = np.array([self.size/2, self.size/2]) if self.determined_start_pos else np.random.uniform(low=0, high=self.size, size=(2,))
		self.goals_found = 0
		self.time_step = 0
		return self._get_cur_state()

	def step(self, action):
		action = DIR[action]
		self.pos = np.clip(self.pos + action, 0, self.size)
		reward = 0
		done = False
		self.time_step += 1
		for i in range(self.num_goal_objects):
			# goal_state = self.goal_states[i]
			goal_state = self.goal_set[i]
			if np.linalg.norm(self.pos - goal_state) < self.goal_radius: 
				# reward += self.goal_rew
				reward += self.goal_rewards[i]
				self.goals_found += 1
		done = (self.goals_found >= self.num_goals) or (self.time_step >= self.max_time_step)
		return self._get_cur_state(), reward, done, {}

	def render(self, mode='human', close=False):
		print(f"Position: {self.pos}")
		print(f"Goal Position: {self.goal_state}, Radius: {self.goal_radius}, Reward: {self.goal_rew}")



if __name__ == '__main__':
	env = TwoDimNavComplexObjCentric(max_time_step=50)
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









