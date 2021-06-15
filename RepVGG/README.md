# RepVGG

## Usage
### Train & Test
```
python3 main.py --model "A0"
```
Other models: A1, A2, B0, B1, B2, B3

### References
* [RepVGG Paper](https://arxiv.org/pdf/2101.03697.pdf)
* [Tensorflow Documentation on Custom Layers](https://www.tensorflow.org/tutorials/customization/custom_layers)
* 
### Contributed by:
* [Aditya Tanwar](https://github.com/cliche-niche/)

## Summary

### Introduction
A simple but powerful architecture of convolutional neural network, which has a VGG-like inferencetime body composed of nothing but a stack of 3x3 convolution and ReLU, while the 
training-time model has a multi-branch topology.

Such decoupling of the training time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG.

### Architecture

### Reparameterization
