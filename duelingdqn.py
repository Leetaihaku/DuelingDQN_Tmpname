import torch
import numpy as np
import gym
import copy

from collections import namedtuple
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

EPISODES =
STEPS =
NODES =
LEARNING_RATE =
NUM_STATES =
NUM_ACTIONS =

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

    def __len__(self):

    def save(self):

    def sampling(self):

class