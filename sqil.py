import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import gym
import numpy as np

import os.path
import pickle

"""
TODO LIST:
- Train on only one path over and over again
- Try activation function
    - Relu
    - Sigmoid
    - LeakyRelu
    - Gelu
- Change the network architecture
- Run Q network on expert example
"""

from BC_train import ExpertDemos

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = 'cpu' # TODO change back

"""Old Memory class using deque"""
# class Memory(object):
#     def __init__(self, memory_size: int) -> None:
#         self.memory_size = memory_size
#         self.buffer = deque(maxlen=self.memory_size)

#     def add(self, experience) -> None:
#         self.buffer.append(experience)

#     def size(self):
#         return len(self.buffer)

#     def sample(self, batch_size: int, continuous: bool = True):
#         if batch_size > len(self.buffer):
#             batch_size = len(self.buffer)
#         if continuous:
#             rand = random.randint(0, len(self.buffer) - batch_size)
#             return [self.buffer[i] for i in range(rand, rand + batch_size)]
#         else:
#             indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
#             return [self.buffer[i] for i in indexes]

#     def clear(self):
#         self.buffer.clear()

#     def save(self, path):
#         b = np.asarray(self.buffer)
#         print(b.shape)
#         np.save(path, b)

#     def load(self, path):
#         b = np.load(path+'.npy', allow_pickle=True)
#         assert(b.shape[0] == self.memory_size)

#         for i in range(b.shape[0]):
#             self.add(b[i])

""" New Memory class with np.array """
class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path):
        """
        Load from a .npy file specified by `path`.
        """
        b = np.load(path+'.npy', allow_pickle=True)
        assert(b.shape[0] == self.memory_size)

        for i in range(b.shape[0]):
            self.add(b[i])
    
    def load_from_np(self, arr: np.array):
        """
        Load from a numpy array.
        """
        assert(arr.shape[0] == self.memory_size)
        for i in range(arr.shape[0]):
            self.add(arr[i])

class SoftQNetwork(nn.Module):
    def __init__(self, num_actions):
        super(SoftQNetwork, self).__init__()
        self.alpha = 4
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU() # shell 1
        self.sig = nn.Sigmoid() # shell 2
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, num_actions)
        
    def forward(self, x):
        x = self.relu(self.fc1(x)) # shell 1
        x = self.relu(self.fc2(x)) # shell 1
        # x = self.sig(self.fc1(x)) # shell 2
        # x = self.sig(self.fc2(x)) # shell 2
        x = self.fc3(x)
        return x

    def getV(self, q_value):
        """
        Uses the log-sum-exp trick to achieve numerical stability and prevent
        overflow. See: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        """
        normalization_constant, _ = torch.max(q_value / self.alpha, dim=1, keepdim=True)
        v = normalization_constant + torch.log(torch.sum(torch.exp(
            q_value/self.alpha - normalization_constant), dim=1, keepdim=True))
        v *= self.alpha
        return v
        
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print('state : ', state)
        with torch.no_grad():
            print('state = ', state)
            q = self.forward(state)
            # q = torch.clamp(q, max=340)
            v = self.getV(q).squeeze()
            print('q & v', q, v)
            dist = torch.exp((q-v)/self.alpha)
            # print(dist)
            dist = dist / torch.sum(dist)
            print('dist =', dist)
            c = Categorical(dist) # print dist in examples, and lookup torch.distributions
            a = c.sample()
        return a.item()

def initialize_expert_replay_buffer(expert_demos: ExpertDemos,
    expert_replay_buffer: Memory, num_demos: int):
    rewards = torch.ones(expert_demos.obs.shape[0])
    data_list = [expert_demos.obs.tolist(),
                expert_demos.next_obs.tolist(), 
                expert_demos.acts.tolist(), rewards.tolist(), 
                expert_demos.dones.tolist()]
    demos = list(zip(*data_list))
    replay_buffer = np.array(demos[:num_demos], dtype=object)
    expert_replay_buffer.load_from_np(replay_buffer)

# TODO: change this function to match newly update ExpertDemos constructor
def create_expert_demos(generate_new_data: bool, expert_demos_filename: str,
    grid_size=10, num_paths=5000):
    """
    Creates expert demos based on whether new data is requested and whether the
    pickled file for the expert demos exists.
    """
    if not generate_new_data and os.path.isfile(expert_demos_filename):
        print('No request for new data and data file exists!\nLoading data...')
        with open(expert_demos_filename, 'rb') as f:
            expert_demos = pickle.load(f)
    else:
        # TODO change ExpertDemos constructor to be based on num_demos rather than
        # paths to solve
        expert_demos = ExpertDemos(size=grid_size, num_paths=num_paths)
        with open(expert_demos_filename, 'wb') as f:
            pickle.dump(expert_demos, f)
    return expert_demos

if __name__ == "__main__":
    #TODO get rid of seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    grid_size = 10
    num_paths = 5000
    expert_demos_filename = f'data/expert_demos({grid_size},{num_paths}).pkl'
    
    # change if you want to guarantee the generation of new data
    generate_new_data = False 
    expert_demos = create_expert_demos(generate_new_data, expert_demos_filename,
                    grid_size, num_paths)

    
    # env = gym.make('CartPole-v0')
    env = expert_demos.env
    num_actions = env.action_space.n # get number of actions
    onlineQNetwork = SoftQNetwork(num_actions).to(device)
    targetQNetwork = SoftQNetwork(num_actions).to(device)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

    learning_rate = 1e-4 # Original learning rate is 1e-4
    
    optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=learning_rate)

    GAMMA = 0.99
    REPLAY_MEMORY = 50000
    BATCH = 16
    UPDATE_STEPS = 4 # 4

    expert_memory_replay = Memory(REPLAY_MEMORY//2)
    initialize_expert_replay_buffer(expert_demos, expert_memory_replay, REPLAY_MEMORY // 2)
    # expert_memory_replay.load('expert_replay')
    # import pdb; pdb.set_trace()
    online_memory_replay = Memory(REPLAY_MEMORY//2)
    writer = SummaryWriter('logs/sqil')

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    saved_policy_file_name = 'sqil-policy.para'

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        for time_steps in range(200):
            # if time_steps == 40 and epoch == 21:
            #     import pdb; pdb.set_trace()
            print('time_steps =', time_steps, 'ep =', epoch)
            action = onlineQNetwork.choose_action(state)
            print('action =', action)
            next_state, reward, done, _ = env.step(action)
            print('next state =', next_state, '\n')
            episode_reward += reward
            online_memory_replay.add((state, next_state, action, 0., done))
            # import pdb; pdb.set_trace() # TODO remove
            if online_memory_replay.size() > 1280:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % UPDATE_STEPS == 0:
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

                online_batch = online_memory_replay.sample(BATCH//2, False)
                online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = zip(*online_batch)

                online_batch_state = torch.FloatTensor(online_batch_state).to(device)
                online_batch_next_state = torch.FloatTensor(online_batch_next_state).to(device)
                online_batch_action = torch.FloatTensor(online_batch_action).unsqueeze(1).to(device)
                online_batch_reward = torch.FloatTensor(online_batch_reward).unsqueeze(1).to(device)
                online_batch_done = torch.FloatTensor(online_batch_done).unsqueeze(1).to(device)

                expert_batch = expert_memory_replay.sample(BATCH//2, False)
                # expert_batch format: 
                # [(s1, ns1, a1, r1, d1), (s2, ns2, a2, r2, d2), ...]
                expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = zip(*expert_batch)

                expert_batch_state = torch.FloatTensor(expert_batch_state).to(device)
                expert_batch_next_state = torch.FloatTensor(expert_batch_next_state).to(device)
                expert_batch_action = torch.FloatTensor(expert_batch_action).unsqueeze(1).to(device)
                expert_batch_reward = torch.FloatTensor(expert_batch_reward).unsqueeze(1).to(device)
                expert_batch_done = torch.FloatTensor(expert_batch_done).unsqueeze(1).to(device)

                batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
                batch_next_state = torch.cat([online_batch_next_state, expert_batch_next_state], dim=0)
                batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
                batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
                batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)


                with torch.no_grad():
                    next_q = targetQNetwork(batch_next_state)
                    # next_q = torch.clamp(next_q, max=340)
                    print('next_q =', next_q)
                    next_v = targetQNetwork.getV(next_q)
                    print('next_v =', next_v)
                    y = batch_reward + (1 - batch_done) * GAMMA * next_v

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss.item(), global_step=learn_steps)
            
            if done:
                print(f'Done! time_steps = {time_steps}')
                break
            
            state = next_state
        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        if epoch % 10 == 0:
            # import pdb; pdb.set_trace()
            torch.save(onlineQNetwork.state_dict(), saved_policy_file_name)
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))




