import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
from misc import progress_bar

# LeNet model definition
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# PGD adversarial attack
def pgd_attack(model, x, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(x, requires_grad=True)

    for t in range(num_iter):
        loss = F.cross_entropy(model(x + delta), y)
        loss.backward()

        # Use gradient ascent to maximize the loss
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    x_adv = x + delta.detach()
    x_adv = torch.clamp(x_adv, 0, 1)  # Clip the perturbations to maintain valid pixel values
    return x_adv

# Adversarial training function
def train_adversarial(model, train_loader, epsilon, alpha, num_iter, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"\n===> epoch: {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Generate adversarial examples using PGD
            x_adv = pgd_attack(model, data, target, epsilon, alpha, num_iter)

            # Perform adversarial training
            model.zero_grad()
            output = model(x_adv)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

            # if batch_idx % 100 == 0:
            #     print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

# Training the robust LeNet model using PGD adversarial training
lenet_model = LeNet()
train_adversarial(lenet_model, train_loader, epsilon=8/255, alpha=2/255, num_iter=1, num_epochs=30, lr=0.001)
