import torch
import numpy as np
from torch import nn

class GenerativeModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GenerativeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2) # two motors


    def forward(self, x):
        inp = self.fc1(x)
        act = torch.nn.functional.sigmoid(inp)
        hid = self.fc2(act)
        out = self.fc3(hid)
        return hid, out

    def train(self, x):
        pass


def main():
    model = GenerativeModel(2, 4)
    output = model.forward(torch.tensor([1.0, 1.0]))
    print(output)



if __name__ == "__main__":
    main()
