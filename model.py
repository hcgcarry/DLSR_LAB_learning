
import torch.nn as nn
class Net(nn.Module):

    # define the layers
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(4, 4, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(4, 4, padding=1)
        '''
        self.fc1 = nn.Linear(16 * 53 * 53, 8192)
        self.fc2 = nn.Linear(8192, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 84)
        self.fc5 = nn.Linear(84, 10)
        '''
        self.fc1 = nn.Linear(16 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    # concatenate these layers
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 14 * 14)
        # x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.fc3(x)
        return x
