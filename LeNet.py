import torch
import torch.nn as nn
import torch.nn.functional as func

from noise_operator.config import NoNoiseConfig
from noise_operator.operators import NoiseOperator


class LeNet(nn.Module):
    def __init__(self, noise_config=NoNoiseConfig(),
                 epsilon=8/255, alpha=2/255, num_iter=1):
        super(LeNet, self).__init__()

        self._epsilon = epsilon
        self._alpha = alpha
        self._num_iter = num_iter

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Makes some noise x
        self._noise_operater = NoiseOperator(noise_config)

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
    
    def pgd_attack(self, x, y):
        perturbation = torch.zeros_like(x, requires_grad=True)

        for _ in range(self._num_iter):
            x_adv = x + perturbation
            logits = self.forward(x_adv)
            loss = func.cross_entropy(logits, y)
            loss.backward()

            # Use gradient ascent to maximize the loss w.r.t. perturbation
            with torch.no_grad():
                perturbation.data = (perturbation + self._alpha * perturbation.grad.detach().sign()).clamp(-self._epsilon, self._epsilon)

            # Clear the gradients of perturbation in preparation for the next iteration
            perturbation.grad.zero_()

        # After all iterations, we detach the perturbation tensor from the computation graph
        # to obtain the final adversarial example
        x_adv = x + perturbation.detach()
        x_adv = torch.clamp(x_adv, 0, 1)  # Clip the perturbations to maintain valid pixel values
        return x_adv

