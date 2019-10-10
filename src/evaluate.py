from models.resnet import *

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import os

from data.dataloader import load_data


from collections import OrderedDict

# from utils.watcher import ActivationWatcher

# from utils.watcher import ActivationWatcher as ActivationWatcherResNet
import argparse

# parser = argparse.ArgumentParser(description='And the bit goes down: Revisiting the quantization of neural networks')
# parser.add_argument('--block', default='all', type=str,
#                     help='Block to quantize (if all, quantizes whole network)')


import architectures.cifar as models

# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# print(model_names)
net = models.__dict__['wrn'](
                    num_classes=100,
                    depth=28,
                    widen_factor=10,
                    dropRate=0.3,
                )

# for k in net.state_dict():
#     print(k)
print(net.eval())


"""
LOAD CIFAR100
"""
trainloader, testloader = load_data()

"""
Define a Loss function
"""

criterion = nn.CrossEntropyLoss()

# checkpoint = torch.load('model_best.pth.tar')

# state_dict = checkpoint['state_dict']

# updated_state_dict = OrderedDict()
# for k ,v in state_dict.items():
#     updated_state_dict[k[7:]] = v
#     # print(k[6:])

# net.load_state_dict(updated_state_dict)


# PATH = "./models/trained"
# net = torch.load(os.path.join(PATH, "resnet18.pth"))

# print(net)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)





# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # net.to(device)
# # optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# class_correct = list(0. for i in range(100))
# class_total = list(0. for i in range(100))
# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data[0].to(device), data[1].to(device)
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(100):
#     print('Accuracy of %5s : %2d %%' % (
#         i, 100 * class_correct[i] / class_total[i]))

