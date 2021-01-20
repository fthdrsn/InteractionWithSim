import airsim
import cv2
import numpy as np
import os
import cv2
from model import CNNNet
import torch
import time
from AirsimAPI import *
from utils import *
from DDPGmodel import *
from ddpg import *
import random
from collections import namedtuple

TRAIN=True
# total length of chosen observation states
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=AirsimEnv()
    hyperparams = {
        "lrvalue": 5e-2,
        "lrpolicy": 1e-2,
        "gamma": 0.97,
        "episodes": 9000,
        "buffersize": 100000,
        "tau": 1e-2,
        "batchsize": 64,
        "start_sigma": 0.3,
        "end_sigma": 0,
        "sigma_decay_len": 15000,
        "theta": 0.15,
        "maxlength": 5000,
        "clipgrad": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)
    agent = DDPGagent( hyprm,(3,12), device=device)
    noise = OUNoise(env.action_space, hyprm)
    agent.to(device)
    step_counter = 0
    best_reward = 0
    avg_reward=[]
    for eps in range(hyprm.episodes):

        state=env.reset()
        # time.sleep(1)
        episode_reward = 0
        for i in range(hyprm.maxlength):
            step_counter += 1
            action = agent.get_action(state)
            if TRAIN:
                 action = noise.get_action(action, step_counter)
            a_1 = np.clip(action[0], -1, 1)
            a_2 = np.clip(action[1], 0, 1)
            a_3 = np.clip(action[2], 0, 1)*0

            action = np.array([a_1, a_2, a_3])

            next_state, reward, done= env.step(action)
            episode_reward+=reward
            agent.memory.push(state,action,reward,next_state,done)
            if TRAIN:
                if len(agent.memory)>hyprm.batchsize:
                    agent.update(hyprm.batchsize)
            if done:
                break

            state=next_state
        avg_reward.append(episode_reward)
        avearage_reward = np.mean(avg_reward[-20:])
        print(f"Episode:{eps}  Average Reward: {avearage_reward}")
