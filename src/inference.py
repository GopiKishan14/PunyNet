import os
import time
import math
import argparse
from operator import attrgetter
from bisect import bisect_left
from models.resnet import *

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn

import models
from data import load_data
from optim import CentroidSGD
from quantization import PQ
from utils.training import finetune_centroids, evaluate
from utils.watcher import ActivationWatcher
from utils.dynamic_sampling import dynamic_sampling
from utils.statistics import compute_size
from utils.utils import centroids_from_weights, weight_from_centroids

parser = argparse.ArgumentParser(description='And the bit goes down: Revisiting the quantization of neural networks')


parser.add_argument('--block', default='all', type=str,
                    help='Block to quantize (if all, quantizes whole network)')

parser.add_argument('--n-iter', default=100, type=int,
                    help='Number of EM iterations for quantization')
parser.add_argument('--n-activations', default=1024, type=int,
                    help='Size of the batch of activations to sample from')

parser.add_argument('--block-size-cv', default=9, type=int,
                    help='Quantization block size for 3x3 standard convolutions')
parser.add_argument('--block-size-pw', default=4, type=int,
                    help='Quantization block size for 1x1 convolutions')
parser.add_argument('--block-size-fc', default=4, type=int,
                    help='Quantization block size for fully-connected layers')

parser.add_argument('--n-centroids-cv', default=256, type=int,
                    help='Number of centroids')
parser.add_argument('--n-centroids-pw', default=256, type=int,
                    help='Number of centroids for pointwise convolutions')
parser.add_argument('--n-centroids-fc', default=2048, type=int,
                    help='Number of centroids for classifier')

parser.add_argument('--n-centroids-threshold', default=4, type=int,
                    help='Threshold for reducing the number of centroids')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='For empty cluster resolution')

parser.add_argument('--batch-size', default=64, type=int,
                    help='Batch size for fiuetuning steps')
parser.add_argument('--n-workers', default=0, type=int,
                    help='Number of workers for data loading')

parser.add_argument('--finetune-centroids', default=1000, type=int,
                    help='Number of iterations for layer-wise finetuning of the centroids')
parser.add_argument('--lr-centroids', default=0.05, type=float,
                    help='Learning rate to finetune centroids')
parser.add_argument('--momentum-centroids', default=0.9, type=float,
                    help='Momentum when using SGD')
parser.add_argument('--weight-decay-centroids', default=1e-4, type=float,
                    help='Weight decay')

parser.add_argument('--finetune-whole', default=5000, type=int,
                    help='Number of iterations for global finetuning of the centroids')
parser.add_argument('--lr-whole', default=0.01, type=float,
                    help='Learning rate to finetune classifier')
parser.add_argument('--momentum-whole', default=0.9, type=float,
                    help='Momentum when using SGD')
parser.add_argument('--weight-decay-whole', default=1e-4, type=float,
                    help='Weight decay')
parser.add_argument('--finetune-whole-epochs', default=1, type=int,
                    help='Number of epochs for global finetuning of the centroids')
parser.add_argument('--finetune-whole-step-size', default=3, type=int,
                    help='Learning rate schedule for global finetuning of the centroids')

parser.add_argument('--restart', default='./', type=str,
                    help='Already stored centroids')
parser.add_argument('--save', default='', type=str,
                    help='Path to save the finetuned models')

import os


def main():
    # get arguments
    global args
    args = parser.parse_args()
    args.block = '' if args.block == 'all' else args.block


    PATH = "./models/trained"
    student = torch.load(os.path.join(PATH, "resnet18_2.pth"))
    teacher = torch.load(os.path.join(PATH, "resnet18_2.pth"))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student.to(device)

    teacher.to(device)



    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # layers to quantize (we do not quantize the first 7x7 convolution layer)
    watcher = ActivationWatcher(student)
    layers = [layer for layer in watcher.layers[1:] if args.block in layer]
    # layers = [layer for layer in watcher.layers if args.block in layer]

    # data loading code
    train_loader, test_loader = load_data(batch_size=args.batch_size, nb_workers=args.n_workers)

    # parameters for the centroids optimizer
    opt_centroids_params_all = []

    # book-keeping for compression statistics (in MB)
    size_uncompressed = compute_size(student)
    size_index = 0
    size_centroids = 0
    size_other = size_uncompressed


    t1 = time.time()

    top_1 = evaluate(test_loader, student, criterion)
    print('Time taken validate 10,000 samples : {}s'.format(time.time() - t1))
    # scheduler.step()
    print('Top1 acc of teacher : {:.2f}'.format(top_1))


    # Step 1: iteratively quantize the network layers (quantization + layer-wise centroids distillation)
    print('Loading Quantized network')
    t = time.time()
    top_1 = 0

    for layer in layers:
        #  gather input activations
        n_iter_activations = math.ceil(args.n_activations / args.batch_size)
        watcher = ActivationWatcher(student, layer=layer)
        in_activations_current = watcher.watch(train_loader, criterion, n_iter_activations)
        in_activations_current = in_activations_current[layer]

        # get weight matrix and detach it from the computation graph (.data should be enough, adding .detach() as a safeguard)
        M = attrgetter(layer + '.weight.data')(student).detach()
        sizes = M.size()
        is_conv = len(sizes) == 4

        # get padding and stride attributes
        padding = attrgetter(layer)(student).padding if is_conv else 0
        stride = attrgetter(layer)(student).stride if is_conv else 1
        groups = attrgetter(layer)(student).groups if is_conv else 1

        # block size, distinguish between fully connected and convolutional case
        if is_conv:
            out_features, in_features, k, _ = sizes
            block_size = args.block_size_cv if k > 1 else args.block_size_pw
            n_centroids = args.n_centroids_cv if k > 1 else args.n_centroids_pw
            n_blocks = in_features * k * k // block_size
        else:
            k = 1
            out_features, in_features = sizes
            block_size = args.block_size_fc
            n_centroids = args.n_centroids_fc
            n_blocks = in_features // block_size

        # clamp number of centroids for stability
        powers = 2 ** np.arange(0, 16, 1)
        n_vectors = np.prod(sizes) / block_size
        idx_power = bisect_left(powers, n_vectors / args.n_centroids_threshold)
        n_centroids = min(n_centroids, powers[idx_power - 1])

        # compression rations
        bits_per_weight = np.log2(n_centroids) / block_size

        # number of bits per weight
        size_index_layer = bits_per_weight * M.numel() / 8 / 1024 / 1024
        size_index += size_index_layer

        # centroids stored in float16
        size_centroids_layer = n_centroids * block_size * 2 / 1024 / 1024
        size_centroids += size_centroids_layer

        # size of non-compressed layers, e.g. BatchNorms or first 7x7 convolution
        size_uncompressed_layer = M.numel() * 4 / 1024 / 1024
        size_other -= size_uncompressed_layer

        # number of samples
        n_samples = dynamic_sampling(layer)

        # print layer size
        print('Quantized layer: {}, size: {}, n_blocks: {}, block size: {}, ' \
              'centroids: {}, bits/weight: {:.2f}, compressed size: {:.2f} MB'.format(
               layer, list(sizes), n_blocks, block_size, n_centroids,
               bits_per_weight, size_index_layer + size_centroids_layer))

        # quantizer
        quantizer = PQ(in_activations_current, M, n_activations=args.n_activations,
                       n_samples=n_samples, eps=args.eps, n_centroids=n_centroids,
                       n_iter=args.n_iter, n_blocks=n_blocks, k=k,
                       stride=stride, padding=padding, groups=groups)

        if len(args.restart) > 0:
            # do not quantize already quantized layers
            try:
                # load centroids and assignments if already stored
                quantizer.load(args.restart, layer)
                centroids = quantizer.centroids
                assignments = quantizer.assignments

                # quantize weight matrix
                M_hat = weight_from_centroids(centroids, assignments, n_blocks, k, is_conv)
                attrgetter(layer + '.weight')(student).data = M_hat
                quantizer.save(args.save, layer)

                # optimizer for global finetuning
                parameters = [p for (n, p) in student.named_parameters() if layer in n and 'bias' not in n]
                centroids_params = {'params': parameters,
                                    'assignments': assignments,
                                    'kernel_size': k,
                                    'n_centroids': n_centroids,
                                    'n_blocks': n_blocks}
                opt_centroids_params_all.append(centroids_params)

                # proceed to next layer
                print('codebook loaded, proceeding to next layer\n')
                continue

            # otherwise, quantize layer
            except FileNotFoundError:
                print('Quantize layer first')


    # End of compression + finetuning of centroids
    size_compressed = size_index + size_centroids + size_other
    print('Non-compressed teacher model: {:.2f}MB, compressed student model ' \
          '(indexing + centroids + other): {:.2f}MB + {:.2f}MB + {:.2f}MB = {:.2f}MB, compression ratio: {:.2f}x\n'.format(
          size_uncompressed, size_index, size_centroids, size_other, size_compressed, size_uncompressed / size_compressed))

    # Step 3: finetune whole network
    print('Validating whole network')
    t = time.time()

    # custom optimizer
    optimizer_centroids_all = CentroidSGD(opt_centroids_params_all, lr=args.lr_whole,
                                      momentum=args.momentum_whole,
                                      weight_decay=args.weight_decay_whole)

    # standard training loop
    n_iter = args.finetune_whole
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_centroids_all, step_size=args.finetune_whole_step_size, gamma=0.1)

    # for epoch in range(args.finetune_whole_epochs):
    student.train()
    finetune_centroids(train_loader, student, teacher, criterion, optimizer_centroids_all, n_iter=n_iter)
    t1 = time.time()

    top_1 = evaluate(test_loader, student, criterion)
    print('Time taken validate 10,000 samples : {}s'.format(time.time() - t1))
    scheduler.step()
    print('Top1 acc: {:.2f}'.format(top_1))


    print('Total parameters: {}'.format(sum(p.numel() for p in student.parameters() if p.requires_grad)))
    # print('Total parameters: {}'.format(sum(p.numel() for p in student.parameters())))


if __name__ == '__main__':
    main()
