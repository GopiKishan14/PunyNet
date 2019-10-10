from models.resnet import *

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import os

import architectures.cifar as models

net = models.__dict__['wrn'](
                    num_classes=100,
                    depth=28,
                    widen_factor=10,
                    dropRate=0.3,
                )

print(net.eval())
# PATH = "./models/trained"
# net = torch.load(os.path.join(PATH, "resnet18.pth"))


# net.load_state_dict(torch.load("./compressed/state.pth"))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_parameters(net))

def print_layers(model):
	for layer in model.modules():
		print(layer)

print(count_parameters(net))