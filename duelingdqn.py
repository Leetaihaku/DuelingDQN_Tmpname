import torch
import numpy as np
import gym
import copy
import random

from collections import namedtuple
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam

EPISODES = 1000
STEPS = 200
NODES = 32
LEARNING_RATE = 0.05
NUM_STATES = 3
NUM_ACTIONS = 1
CAPACITY = 10000
BATCH_SIZE = 32
EPSILON = 1
EPSILON_DISCOUNT = 0.001
ERROR_EPSILON = 0.0001
REWARD_DISCOUNT = 0.9

DATA = namedtuple('DATA', ('state', 'action', 'reward', 'next_reward', 'done'))

#신경망
class NeuralNet(nn.Module):
    def __init__(self, Lin, Lmid, Lout):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(Lin, Lmid)
        self.fc2 = nn.Linear(Lmid, Lmid)

        self.fc3_adv = nn.Linear(Lmid, Lout)
        self.fc3_v = nn.Linear(Lmid, 1)

    def forward(self, x):
        ac1 = F.relu(self.fc1(x))
        ac2 = F.relu(self.fc2(ac1))

        adv = self.fc3_adv(ac2)
        val = self.fc3_v(ac2)

        output = val + adv
        return output

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

        indexes = []
        idx = 0
        tmp_sigma_error_abs = 0
        for rand_num in rand_list:
            while tmp_sigma_error_abs < rand_num:
                tmp_sigma_error_abs += abs(self.memory[idx]) + ERROR_EPSILON
                idx += 1

            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_error(self, updated_error):
        self.memory = updated_error

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

class Brain:
    def __init__(self):
        self.q = NeuralNet(NUM_STATES, NODES, NUM_ACTIONS)
        self.tq = NeuralNet(NUM_STATES, NODES, NUM_ACTIONS)
        self.optimizer = Adam(self.q.parameters(), lr=LEARNING_RATE)
        self.db = DB()
        self.error = ErrorDB()
        self.epsilon = EPSILON

    def action_order(self, state):
        if self.epsilon - EPSILON_DISCOUNT < random.uniform(0, 1):
            state = torch.tensor(state).float()
            with torch.no_grad():
                action = self.q.forward(state)
        else:
            action = torch.tensor([random.uniform(-2, 2)])
        return action

    def update_q(self, episode):
        #print(self.q.fc1.weight)
        if self.db.__len__() < BATCH_SIZE:
            return

        if episode < 30:
            batch = self.db.sampling()
        else:
            batch = self.error.prioritized_index()
            batch = [self.db.memory[n] for n in batch]
        batch = DATA(*zip(*batch))
        state_serial = batch.state
        #action_serial = batch.action
        reward_serial = batch.reward
        next_state_serial = batch.next_reward
        done_serial = batch.done

        state_serial = torch.stack(state_serial).float()
        #action_serial = torch.stack(action_serial).float()
        reward_serial = torch.stack(reward_serial).float().reshape([BATCH_SIZE, 1])
        next_state_serial = torch.stack(next_state_serial).float()
        done_serial = torch.stack(done_serial).reshape([BATCH_SIZE, 1])

        self.q.eval()
        self.tq.eval()
        next_max_idx = self.q.forward(next_state_serial).max(1)[1].reshape([BATCH_SIZE, 1])
        next_by_tq = torch.gather(self.tq.forward(next_state_serial), 1, next_max_idx)
        target_q_serial = reward_serial + REWARD_DISCOUNT * (~done_serial) * next_by_tq

        self.q.train()
        q_serial = torch.gather(self.q.forward(state_serial), 1, next_max_idx)
        loss = F.smooth_l1_loss(q_serial, target_q_serial)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_tq(self):
        self.tq = copy.deepcopy(self.q)

class Agent:
    def __init__(self):
        self.brain = Brain()

    def action_request(self, state):
        return self.brain.action_order(state)

    def save_to_db(self, state, action, reward, next_state, done):
        self.brain.db.save(state, action, reward, next_state, done)

    def save_to_errordb(self, error):
        self.brain.error.save(error)

    def train(self, episode):
        self.brain.update_q(episode)

    def update(self):
        self.brain.update_tq()

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent()
    scores = []
    for E in range(EPISODES):
        print('ep', E)
        state = env.reset()
        score = 0
        for S in range(STEPS):
            if E % 100 == 0:
                env.render()
            action = agent.action_request(state)
            next_state, reward, done, info = env.step(np.array(action))
            if next_state[0] >= 0 and next_state[1] >= 0 and next_state[2] >= 1:
                if next_state[0] < state[0] and next_state[1] < next_state[1] and next_state[2] > next_state[2]:
                    reward += 1
            agent.save_to_db(torch.tensor(state), torch.tensor(action), torch.tensor(reward), torch.tensor(next_state), torch.tensor(done))
            agent.save_to_errordb(0)
            agent.train(E)
            if done:
                print('EPISODES : ', E, 'STEPS : ', S, 'SCORES : ', score, 'EPSILON : ', agent.brain.epsilon)
                agent.brain.epsilon -= EPSILON_DISCOUNT
                break
            else:
                state = next_state
                score += reward
        agent.update()
        scores.append(score)
    env.close()