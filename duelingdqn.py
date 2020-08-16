import torch
import numpy as np
import gym
import copy
import random

from collections import namedtuple
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

EPISODES = 1000
STEPS = 200
NODES = 32
LEARNING_RATE = 0.01
NUM_STATES = 1##
NUM_ACTIONS = 1##
CAPACITY = 10000
BATCH_SIZE = 32
EPSILON = 1
ERROR_EPSILON = 0.0001


DATA = namedtuple('DATA', ('state', 'action', 'reward', 'next_reward', 'done'))

#신경망
class NeuralNet:
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, NODES)
        self.fc2 = nn.Linear(NODES, NODES)

        self.fc3_adv = nn.Linear(NODES, NUM_ACTIONS)
        self.fc3_v = nn.Linear(NODES, 1)

    def forward(self, x):
        ac1 = F.relu(self.fc1(x))
        ac2 = F.relu(self.fc2(ac1))

        adv = self.fc3_adv(ac2)
        val = self.fc3_v(ac2).expand(-1, adv.size(1))

        #bias 감쇄(adv 평균감산)
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output

class PrioritizedDB:
        def __init__(self):
            self.db = DB()
            self.errordb =ErrorDB()







class ErrorDB:
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def __len__(self):
        return len(self.memory)

    def save(self, error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = error
        self.index = (self.index+1) % self.capacity

    def prioritized_index(self):
        sigma_error_abs = np.sum(np.absolute(self.memory))
        sigma_error_abs += ERROR_EPSILON * self.__len__()

        rand_list = np.random.uniform(0, sigma_error_abs, BATCH_SIZE)
        rand_list = np.sort(rand_list)





    def update_error(self):



class DB:
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def __len__(self):
        return len(self.memory)

    def save(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = DATA(state, action, reward, next_state, done)
        self.index = (self.index+1) % self.capacity

    def sampling(self):
        return random.sample(self.memory, BATCH_SIZE)

class