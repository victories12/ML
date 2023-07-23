import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import dill
import json
from datetime import datetime

import argparse

from noise_operator.config import NoNoiseConfig,GaussAddConfig,GaussMulConfig,GaussCombinedConfig

from LeNet import *
from misc import progress_bar


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
"NoNoise":0
"GaussAdd":1
"GaussMul":2
"GaussCombined":3
"""
noise_types = [0,1]

gauss_stds = [0.01,0.03,0.04,0.06,0.075,0.09,0.1,0.15,0.2,0.4,0.6,0.7,0.8,1.0,5.0,10.0,30.0,60.0,90,100.0]

epochs = 30

def main():
    for noise_type in noise_types:
        accuracies = []
        if noise_type==0:
            print(f"\n#### NoiseType:{noise_type}")
            parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
            parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
            parser.add_argument('--epochs', default=epochs, type=int, help='number of epochs tp train for')
            parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
            parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
            parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')

            parser.add_argument('--noiseConfig', default=NoNoiseConfig(), type=bool, help='no noise')
            args = parser.parse_args()

            solver = Solver(args)
            acc = solver.run()
            accuracies.append({"noise_type":noise_type,"acc":acc,"epochs":epochs})

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            with open(f"results_{noise_type}_{timestamp}.json", "w") as f:
                json.dump(accuracies, f)
        
        if noise_type==1:
            for gauss_std in gauss_stds:
                print(f"\n#### NoiseType:{noise_type}   GaussStd:{gauss_std}")
                parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
                parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
                parser.add_argument('--epochs', default=epochs, type=int, help='number of epochs tp train for')
                parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
                parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
                parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
                
                parser.add_argument('--noiseConfig', default=GaussAddConfig(GaussMean=0.0,GaussStd=gauss_std), type=bool, help='gauss add noise')
                args = parser.parse_args()

                solver = Solver(args)
                acc = solver.run()
                accuracies.append({"noise_type":noise_type,"gauss_std":gauss_std,"acc":acc,"epochs":epochs})

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            with open(f"results_{noise_type}_{timestamp}.json", "w") as f:
                json.dump(accuracies, f)


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epochs
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.noise_config = config.noiseConfig

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = LeNet(self.noise_config).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path,pickle_module=dill)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print(f"\n===> epoch: {epoch}/{self.epochs}")
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()
                return accuracy


if __name__ == '__main__':
    main()