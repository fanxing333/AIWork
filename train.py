import time

import torch
import torchvision
from torch import nn, optim
from torchvision.transforms import transforms

from TrainingPipeline.models.AlexNet import AlexNet
from TrainingPipeline.models.SimpleCNN import SimpleCNN


class Train:
    def __int__(self, model, trainset, testset):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.trainset = trainset
        self.testset = testset
        self.criterion = None
        self.optimizer = None
        self.loss_per_epoch = []
        self.train_acc = []
        self.test_acc = []

    def set_criterion(self, criterion):
        self.criterion = criterion.to(self.device)

    def set_optimizer(self, base_optimizer, lr, momentum):
        self.optimizer = base_optimizer(self.model.parameters(), lr=lr, momentum=momentum)

    def set_batchsize(self, batchsize):
        self.batchsize = batchsize

    def start_training(self, epoch_num):
        # start training
        start_time = time.time()

        for epoch in range(epoch_num):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            # 保存 epoch 信息
            self.loss_per_epoch.append(running_loss / 50000)
            self.train_acc.append(self.evaluate_accuracy())
            self.test_acc.append(self.evaluate_accuracy())

            print(f'[{epoch}] loss: {running_loss / 50000:.5f} '
                  f'train error: {self.train_acc[-1]:.3f} '
                  f'test error: {self.test_acc[-1]:.3f}')
            running_loss = 0.0

        end_time = time.time()

    def evaluate_accuracy(self):
        pass


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    T = Train(model=SimpleCNN(), trainset=trainset, testset=testset)
    T.set_criterion(nn.CrossEntropyLoss())
    T.set_optimizer(optim.SGD, lr=0.01, momentum=0.9)