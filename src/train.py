from models.resnet import *

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import os

from data.dataloader import load_data

from collections import OrderedDict



# from __future__ import print_function

# import argparse
# import os
# import shutil
# import time
# import random

# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import architectures.cifar as models

# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))


# print(model_names)
# net = models.__dict__['wrn'](
#                     num_classes=100,
#                     depth=28,
#                     widen_factor=10,
#                     dropRate=0.3,
#                 )

# for k in net.state_dict():
#     print(k)



"""
LOAD CIFAR100
"""
trainloader, testloader = load_data()

"""
Define a Loss function
"""

criterion = nn.CrossEntropyLoss()
"""
Training on GPU
"""
net = resnet18()

print(net.eval())
#For retraining, load the previously trained model
# net = torch.load("resnet18.pth")

# PATH = "./models/trained"
# net = torch.load(os.path.join(PATH, "resnet18.pth"))

# checkpoint = torch.load('model_best.pth.tar')

# state_dict = checkpoint['state_dict']

# updated_state_dict = OrderedDict()
# for k ,v in state_dict.items():
#     updated_state_dict[k[7:]] = v
#     # print(k[6:])

# net.load_state_dict(updated_state_dict)

# print(net)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# optimizer.load_state_dict(checkpoint['optimizer'])#Optimizing parameters

# epoch = checkpoint['epoch']#epoch, can be used to update learning rate, etc.



"""
Other optimizers at https://pytorch.org/docs/stable/optim.html
"""


"""
Train the network on CIFAR-100
"""

epochs = 30
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]        
                   
        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        period = 200
        if i % period == period-1:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / period))
            running_loss = 0.0

print('Finished Training')

PATH = "./models/trained"
torch.save(net, os.path.join(PATH, "renet18.pth"))
print(" Model saved at {}", PATH)


"""
TESTING the model
# """
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(inputs.shape , outputs.shape , labels.shape)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(100):
    print('Accuracy of %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))


# torch.save(net, "resnet18.pth")

# model = torch.load("resnet18.pth")
