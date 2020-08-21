import torch
import gym
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
import copy
import random
from collections import namedtuple
ENV = 'CartPole-v0'
PATH = 'duelingdqn_ori.pth'
EPISODES = 10000
STEPS = 400
TRAIN_START = 1000
NUM_STATES = 4
NUM_ACTIONS = 2
NODES = 16
LEARNING_RATE = 0.01
CAPACITY = 10000
BATCH_SIZE = 32
DISCOUNT = 0.9
EPSILON_DISCOUNT = 0.0001
DATA = namedtuple('DATA', ('state','action','reward','next_state','done'))

class db:
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def save(self, state, action, reward, next_state, done):
        if self.__len__() < CAPACITY:
            self.memory.append(None)
        self.memory[self.index] = DATA(state,action,reward,next_state,done)
        self.index = (self.index+1)%CAPACITY

    def sampling(self):
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)

class neural_network:
    def modeling_nn(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(NUM_STATES, NODES))
        model.add_module('relu1', nn.ReLU())
        model.add_module('drop1', nn.Dropout(p=0.5))
        model.add_module('fc2', nn.Linear(NODES, NODES))
        model.add_module('relu2', nn.ReLU())
        model.add_module('drop2', nn.Dropout(p=0.5))
        model.add_module('fc3', nn.Linear(NODES, NUM_ACTIONS))
        return model

class brain:
    def __init__(self):
        self.db = db()
        self.nn = neural_network()
        self.Q = self.nn.modeling_nn()
        self.TQ = self.nn.modeling_nn()
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)
        self.epsilon = 1.0
        self.epsilon_discount = EPSILON_DISCOUNT

    def action(self,state):
        if random.uniform(0, 1) > self.epsilon:
            state = torch.from_numpy(state).float()
            self.Q.eval()
            with torch.no_grad():
                action = torch.argmax(self.Q(state)).item()
        else:
            action = random.randrange(0, NUM_ACTIONS)
        return action

    def update_q(self):
        batch = self.db.sampling()
        batch = DATA(*zip(*batch))
        state_serial = batch.state
        action_serial = batch.action
        reward_serial = batch.reward
        next_state_serial = batch.next_state
        done_serial = batch.done

        state_serial = torch.tensor(state_serial).float()
        action_serial = torch.tensor(action_serial).long().reshape(BATCH_SIZE, 1)
        reward_serial = torch.tensor(reward_serial).float().reshape(BATCH_SIZE, 1)
        next_state_serial = torch.tensor(next_state_serial).float()
        done_serial = torch.tensor(done_serial).reshape(BATCH_SIZE, 1)

        self.Q.eval()
        self.TQ.eval()
        q_next_idx = self.Q(next_state_serial).max(1)[1]
        tq_next_val = torch.gather(self.TQ(next_state_serial), 1, torch.reshape(q_next_idx, [BATCH_SIZE, 1]))
        tq_next_val = reward_serial + DISCOUNT * tq_next_val * (~done_serial)
        q_val = torch.gather(self.Q(state_serial), 1, torch.reshape(action_serial,[BATCH_SIZE, 1]))

        self.Q.train()
        loss = F.smooth_l1_loss(q_val, tq_next_val)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_tq(self):
        self.TQ = copy.deepcopy(self.Q)

class agent:
    def __init__(self):
        self.brain = brain()

    def save_to_db(self,state, action, reward, next_state, done):
        self.brain.db.save(state, action, reward, next_state, done)

    def action_request(self,state):
        return self.brain.action(state)

    def learn(self):
        self.brain.update_q()

    def update(self):
        self.brain.update_tq()

def reuse():
    print('모델을 선택해주세요')
    print('[1]새 학습하기\t[2]이어 학습하기\t[3]테스트모드')
    answer = input()
    if answer == '1':
        return 'new'
    elif answer == '2':
        return 'old'
    elif answer == '3':
        return 'test'
    else:
        print('입력 값이 상이합니다')
        exit()

def reward_calculator(next_state):
    position = abs(0 - next_state[0])
    angle = abs(0 - next_state[2])
    return -(position + angle)

if __name__ == '__main__':
    env = gym.make(ENV)
    agent = agent()
    answer = reuse()
    if answer != 'new':
        agent.brain.Q.load_state_dict(torch.load(PATH))
        agent.brain.TQ.load_state_dict(torch.load(PATH))
    if answer == 'test':
        '''테스트모드'''
        agent.brain.Q.eval()
        agent.brain.TQ.eval()
        agent.brain.epsilon = 0
        for E in range(EPISODES):
            state = env.reset()
            score = 0
            for S in range(STEPS):
                env.render()
                action = agent.action_request(state)
                next_state, reward, done, _ = env.step(action)
                if done:
                    print('EPISODE : ', E, 'SCORE : ', score, 'EPSILON', agent.brain.epsilon)
                    break
                else:
                    state = next_state
                    score += reward
        env.close()
    else:
        '''훈련모드'''
        for E in range(EPISODES):
            state = env.reset()
            score = 0
            for S in range(STEPS):
                if E%100 == 0 and E!=0:
                    env.render()
                action = agent.action_request(state)
                next_state, reward, done, _ = env.step(action)
                #reward += reward_calculator(next_state)
                agent.save_to_db(state, action, reward, next_state, done)
                if agent.brain.db.__len__() >= TRAIN_START:
                    agent.learn()
                if done:
                    print('EPISODE : ', E, 'SCORE : ', score, 'EPSILON', agent.brain.epsilon)
                    break
                else:
                    state = next_state
                    score += reward
            agent.brain.epsilon -= agent.brain.epsilon_discount
            agent.update()
        print('학습 결과를 저장하시겠습니까? [Y/N]')
        answer = input()
        if answer == 'y' or answer == 'Y':
            torch.save(agent.brain.Q.state_dict(), PATH)
        env.close()



