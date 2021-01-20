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
import random
device="cpu"
env=AirsimEnv()

def process_images(images):
    t1=images.transpose(1,3).transpose(2,3)
    t=t1.reshape(t1.shape[0],-1,t1.shape[2]*t1.shape[3])
    mx,_=t.max(dim=2)
    mn,_ =t.min(dim=2)
    t=(t.float()-mn.unsqueeze(-1))/mx.unsqueeze(-1)
    return t.reshape(t1.shape).to(device)

env.reset()
replay_buffer=Memory(10000)
ddpgActor=Actor((3,12))
ddpgCritic=Critic((3,12))

while True:
    throttle=random.uniform(0,1)
    steer = random.uniform(-1, 1)
    brake = random.uniform(0, 1)
    action=[throttle,steer,brake]
    next_state,reward,done=env.step(action)
    actions=ddpgActor([torch.tensor(next_state[0]).unsqueeze(0).float(),torch.tensor(next_state[1]).unsqueeze(0).float()])
    values=ddpgCritic([torch.tensor(next_state[0]).unsqueeze(0).float(),torch.tensor(next_state[1]).unsqueeze(0).float()],torch.tensor(actions))
    replay_buffer.push(next_state,action,reward,next_state,done)

    if len(replay_buffer)>32:
        k=replay_buffer.sample(16)




    # font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (72, 128)
    # fontScale = 1
    # fontColor = (255, 255, 255)
    # lineType = 2
    # cv2.putText(next_state, str(reward),
    #             bottomLeftCornerOfText,
    #             font,
    #             fontScale,
    #             fontColor,
    #             lineType)
    if done:
         env.reset()
    # cv2.imshow("Center",next_state)
    # cv2.waitKey(10)



# client.enableApiControl(False)
