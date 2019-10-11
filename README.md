# PunyNet
MicroNet: Large-Scale Model Compression Competition.
Our results are for CIFAR-100 dataset.

## 1. Approach
Our method is based on the works of the paper [And the bit goes down: Revisiting the quantization of neural networks](https://arxiv.org/abs/1907.05686) by Facebook Research. Basically, it suggests a compression method based on vector quantization. A student model is optimized using a trained teacher model through a distillation procedure at all layers and a fine-tuning stage.

We are interested in preserving the output of the layer, not its weights. Preserving the weights a layer does not necessarily guarantee preserving its output. In other words, the Frobenius approximation of the weights of a layer is not guaranteed to be the best approximation of the output over some arbitrary domain (in particular
for in-domain inputs).
More precisely, given a batch of B input activations x ∈ R B×C in , we are interested in learning a codebook C that minimizes the difference between the output activations and their reconstructions.

We quantize the network sequentially starting from the lowest layer to the highest layer. We guide the compression of the student network by the non-compressed teacher network, as detailled below.

<b>Learning the codebook</b>  We recover the current input activations of the layer,i.e.the input activations obtained by forwarding a batch of images through the quantized lower layers, and we quantize the current layer using those activations.

<b>Finetuning the codebook</b> We finetune the codewords by distillation (Hinton et al., 2014) using the non-compressed network as the teacher network and the compressed network (up to the current layer) as the student network.  Denoting yt(resp. ys) the output probabilities of the teacher(resp.  student) network, the loss we optimize is the Kullback-Leibler divergence L=KL(ys,yt).Finetuning on codewords is done by averaging the gradients of each subvector assigned to a given codeword.

In a final step,  we globally finetune the codebooks of all the layers to reduce any residual driftsand we update the running statistics of the BatchNorm layers.

## 2. Reproducing the results

Please check the req.txt for environment set up suggested to run the code.
Clone the repository and perform following steps in PunyNet directory.
#### Training resnet18 
```
cd src
python3 train.py
```
#### Quantizing resnet18
```
cd src
python3 quantize.py
```
#### Inferencing with quantized model
```
cd src
python3 inference.py
```

## 3. Results and Calculation

The non-compressed teacher resnet18 achieves Top1 acc 60.24% on training for 30 epoches, while the compressed student resnet18 achieves Top1 acc 53.39% on 9 epoches of finetuning.

non-compressed teacher model: 42.83MB
compressed student model (indexing + centroids + other): 1.22MB + 0.09MB + 0.07MB = 1.39MB
compression ratio: 30.89x

#### score calculation : 
For CIFAR-100, parameter storage and compute requirements is be normalized relative to [WideResNet-28-10](https://arxiv.org/pdf/1605.07146.pdf), which has 36.5M parameters and 10.49B math operations.

Our compressed resnet18 has parameters 11.22M and total math operations = .......

