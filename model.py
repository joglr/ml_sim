import math
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from constants import MAX_WHEEL_SPEED

class GenerativeModel(nn.Module):
    def __init__(self, hidden_size, beam_size, beam_count, lr=100):
        super(GenerativeModel, self).__init__()
        self.data = []
        self.input_layer = nn.Linear(beam_count, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 2) # two motors
        self.BEAM_SIZE = beam_size
        self.BEAM_COUNT = beam_count
        self.optimizer = torch.optim.SGD(self.parameters(), lr)
        self.max_beam_sum = beam_size * beam_count
        self.itters = 0


    def forward(self, x):
        inp = self.input_layer(torch.Tensor(x).float())
        act = torch.nn.functional.sigmoid(inp)

        hid = self.hidden_layer(act)
        act = torch.nn.functional.sigmoid(hid)

        out = self.output_layer(act)
        norm = torch.nn.functional.sigmoid(out)
        return norm

    def train(self, lidar_scans):
        lidar_scans = np.array(lidar_scans) / self.BEAM_SIZE
        out = self.forward(lidar_scans)
        self.optimizer.zero_grad()
        fitness = self.fitness_fun(out, lidar_scans)
        self.data.append(fitness.item())
        if self.itters > 200:
            plt.plot(self.data)
            plt.show()
        fitness.backward()
        self.optimizer.step()
        self.itters += 1
        return out * MAX_WHEEL_SPEED

    def fitness_fun(self, out, lidar_scans):
        wheel_velocity = out
        left_wheel, right_wheel = wheel_velocity


        λ = 1
        fitness = λ * (left_wheel / MAX_WHEEL_SPEED) * (right_wheel / MAX_WHEEL_SPEED) + (np.sum(lidar_scans) / self.max_beam_sum)
        return fitness

