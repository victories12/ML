import torch.nn as nn
import torch.nn.functional as func

from noise_operator.config import NoNoiseConfig,GaussAddConfig,GaussMulConfig,GaussCombinedConfig
from noise_operator.operators import NoiseOperator


class LeNet(nn.Module):
    def __init__(self, default_noise_config=NoNoiseConfig()):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Makes some noise xD
        self.noise_config=default_noise_config
        # self.noise_config=GaussAddConfig
        # self.noise_config=GaussMulConfig
        # self.noise_config=GaussCombinedConfig
        self._noise_operater = NoiseOperator(self.noise_config)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self._noise_operater.forward(x)
        x = func.max_pool2d(x, 2)
        x = self._noise_operater.forward(x)
        x = func.relu(self.conv2(x))
        x = self._noise_operater.forward(x)
        x = func.max_pool2d(x, 2)
        x = self._noise_operater.forward(x)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = self._noise_operater.forward(x)
        x = func.relu(self.fc2(x))
        x = self._noise_operater.forward(x)
        x = self.fc3(x)
        x = self._noise_operater.forward(x)
        return x