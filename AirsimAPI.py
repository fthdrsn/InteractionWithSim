import airsim
import cv2
import numpy as np
import os
import cv2
from model import CNNNet
import torch
import time
from model import CNNNet
from csv_reader import  RoadPoints
from collections import deque
from statistics import mean
import gym
from gym import spaces

class AirsimEnv():
    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.device="cuda"
        self.reward_model=CNNNet().to(self.device)
        self.reward_model.load_state_dict(torch.load("model.dat",map_location=self.device))
        self.reward_model.eval()
        # input space.
        high = np.array([np.inf, np.inf, 1., 1.])
        low = np.array([-np.inf, -np.inf, 0., 0.])
        self.observation_space = spaces.Box(low=low, high=high)
        # action space: [steer, accel, brake]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.default_action = [0.0, 1.0, 0.0]
        # store vehicle speeds
        self.max_speed = 3e5
        self.prev_speed_sample = 40
        self.past_vehicle_speeds = deque([self.max_speed] * self.prev_speed_sample,
                                         maxlen=self.prev_speed_sample)
        self.done=False
        self.lower_speed_limit = 5
        #Convert Airsim image to numpy
    def get_image(self,image):
        image1d = np.fromstring(image.image_data_uint8, dtype=np.uint8)
        image_rgb = image1d.reshape(image.height, image.width, 3)
        return image_rgb

    def process_image(self, image):
        image_trans=np.transpose(image,(2,0,1))
        return image_trans
    # #Normalization and adaptation for torch
    # def process_images(self,images):
    #     t1=images.transpose(1,3).transpose(2,3)
    #     t=t1.reshape(t1.shape[0],-1,t1.shape[2]*t1.shape[3])
    #     mx,_=t.max(dim=2)
    #     mn,_ =t.min(dim=2)
    #     t=(t.float()-mn.unsqueeze(-1))/mx.unsqueeze(-1)
    #     return t.reshape(t1.shape).to(self.device)
    def sim_state_to_env(self,car_state):
        car_states_vec = []
        angular_acc = car_state.kinematics_estimated.angular_acceleration
        car_states_vec.append(angular_acc.x_val)
        car_states_vec.append(angular_acc.y_val)
        car_states_vec.append(angular_acc.z_val)
        angular_vel = car_state.kinematics_estimated.angular_velocity
        car_states_vec.append(angular_vel.x_val)
        car_states_vec.append(angular_vel.y_val)
        car_states_vec.append(angular_vel.z_val)
        linear_acc = car_state.kinematics_estimated.linear_acceleration
        car_states_vec.append(linear_acc .x_val)
        car_states_vec.append(linear_acc .y_val)
        car_states_vec.append(linear_acc .z_val)
        linear_vel = car_state.kinematics_estimated.linear_velocity
        car_states_vec.append(linear_vel.x_val)
        car_states_vec.append(linear_vel.y_val)
        car_states_vec.append(linear_vel.z_val)
        return car_states_vec
    #Take a step in environment
    def step(self,action):
         all_states=[]
         self.car_controls.throttle=action[1]
         self.car_controls.steering = action[0]
         self.car_controls.brake = action[2]
         self.client.setCarControls(self.car_controls)
         car_state = self.client.getCarState()
         state_vec=self.sim_state_to_env(car_state)
         # pos= client.simGetPositionWRTOrigin()
         response = self.client.simGetImages([airsim.ImageRequest("CenterCamera", airsim.ImageType.Scene, False, False)])[0]
         # scene vision image in uncompressed RGB array
         im = self.get_image(response)
         processed_im = self.process_image(im)
         reward,self.done=self.CalculateReward(processed_im,car_state)
         all_states.append(processed_im)
         all_states.append(np.array(state_vec))
         return  all_states,reward,self.done

    # def getDistanceToCenter(position):
    #     min_dist=100000
    #     for points in road_point_locations:
    #         dist=np.sqrt((position.x_val-4255.0-points[0])**2+(position.y_val-20295.0-points[1])**2)
    #         if dist<min_dist:
    #             min_dist=dist
    #     return  min_dist

    def CalculateReward(self,im,car_state):

        prob=self.reward_model(torch.tensor(im).to(self.device).unsqueeze(0).float())
        speed=car_state.speed
        colision_info=self.client.simGetCollisionInfo()
        self.done=colision_info.has_collided
        reward=speed*prob.item()
        self.past_vehicle_speeds.append(speed)
        if self.done:
            reward=-20
        self.past_vehicle_speeds.append(speed * 3.6)  # m/s to km/h
        speed_mean = mean(self.past_vehicle_speeds)
        # print("SpeedMean: ",speed_mean)
        if speed_mean < self.lower_speed_limit:
            print("[SimstarEnv] finish episode bc agent is too slow")
            reward = -20
            self.done = True
        return  reward,self.done

    def reset(self):
        self.past_vehicle_speeds = deque([self.max_speed] * self.prev_speed_sample,
                                         maxlen=self.prev_speed_sample)
        self.done=False
        self.client.reset()
        all_states=[]
        car_state = self.client.getCarState()
        state_vec = self.sim_state_to_env(car_state)
        # pos= client.simGetPositionWRTOrigin()
        response = self.client.simGetImages([airsim.ImageRequest("CenterCamera", airsim.ImageType.Scene, False, False)])[0]
        # scene vision image in uncompressed RGB array
        im = self.get_image(response)
        processed_im = self.process_image(im)
        all_states.append(processed_im)
        all_states.append(np.array(state_vec))
        return all_states




