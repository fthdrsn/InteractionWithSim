import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

class Critic(nn.Module):

    def __init__(self, input_dim, hidden_size=0,output_dim=0 ):
        super().__init__()
        self.input_dim = input_dim[0]
        self.input_dim_sensor = input_dim[1]
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        ##Image encoder
        self.bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 6, 5)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.fc1_im = nn.Linear(16 * 33 * 61, 120)
        ##Sensory data
        self.bn2 = nn.BatchNorm1d(self.input_dim_sensor)
        self.fc1_sensor = nn.Linear(self.input_dim_sensor, 64)
        torch.nn.init.xavier_uniform(self.fc1_sensor.weight)
        ##Action data
        self.bn3 = nn.BatchNorm1d(3)
        self.fc1_action = nn.Linear(3, 64)
        torch.nn.init.xavier_uniform(self.fc1_action.weight)
        ##Fully connected layers
        self.fc1 = nn.Linear(248, 600)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(600, 300)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(300, 3)
        torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x,action):

        image_data = x[0]
        sensor_data = x[1]
        image_data = self.pool(F.relu(self.conv1(image_data)))
        image_data = self.pool(F.relu(self.conv2(image_data)))
        image_data = image_data.reshape(-1, 16 * 33 * 61)
        image_data = self.fc1_im(image_data)
        sensor_data = F.relu(self.fc1_sensor(sensor_data))
        action_data=F.relu(self.fc1_action(action))
        all_data = torch.cat((image_data, sensor_data,action_data), dim=1)
        o1 = F.relu(self.fc1(all_data))
        o2 = F.relu(self.fc2(o1))
        o3 = self.fc3(o2)
        return o3

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_size=0,output_dim=0):
        super().__init__()
        self.input_dim = input_dim[0]
        self.input_dim_sensor = input_dim[1]
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        ##Image encoder
        self.bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 6, 5)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.fc1_im = nn.Linear(16 * 33 * 61, 120)
        ##Sensory data
        self.bn2 = nn.BatchNorm1d(self.input_dim_sensor)
        self.fc1_sensor = nn.Linear(self.input_dim_sensor,64)
        torch.nn.init.xavier_uniform(self.fc1_sensor.weight)
        ##Fully connected layers
        # hidden=torch.cat((self.fc1_im,self.fc1_sensor))
        self.fc1 = nn.Linear(184, 600)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(600, 300)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(300, 3)
        torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        image_data=x[0]
        sensor_data=x[1]
        image_data = self.pool(F.relu(self.conv1(image_data)))
        image_data = self.pool(F.relu(self.conv2(image_data)))
        image_data = image_data.reshape(-1, 16 * 33 * 61)
        image_data=self.fc1_im(image_data)
        sensor_data=F.relu(self.fc1_sensor(sensor_data))
        all_data=torch.cat((image_data,sensor_data),dim=1)
        o1=F.relu(self.fc1(all_data))
        o2=F.relu(self.fc2(o1))
        o3=self.fc3(o2)
        steer = o3[:, 0].reshape(-1, 1)
        accel = o3[:, 1].reshape(-1, 1)
        brake = o3[:, 2].reshape(-1, 1)
        steer = torch.tanh(steer)
        accel = torch.sigmoid(accel).reshape(-1, 1)
        brake = torch.sigmoid(brake).reshape(-1, 1)

        return torch.cat((steer, accel, brake), 1)









