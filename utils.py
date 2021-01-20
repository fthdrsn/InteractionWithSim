import numpy as np
import gym
from collections import deque
import random
import torch.nn as nn

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, params):
        self.mu           = 0.
        self.theta        = params.theta
        self.sigma        = params.start_sigma
        self.max_sigma    = params.start_sigma
        self.min_sigma    = params.end_sigma
        self.decay_period = params.sigma_decay_len
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
        

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    def __len__(self):
        return len(self.buffer)
    def sample(self, batch_size):
        state_im_batch = []
        state_sensor_batch = []
        action_batch = []
        reward_batch = []
        next_state_im_batch = []
        next_state_sensor_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_im_batch.append(state[0])
            state_sensor_batch.append(state[1])
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_im_batch.append(next_state[0])
            next_state_sensor_batch.append(next_state[1])
            done_batch.append(done)
        
        return np.array(state_im_batch),np.array(state_sensor_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_im_batch),np.array(next_state_sensor_batch), np.array(done_batch)

    def __len__(self):
        return len(self.buffer)