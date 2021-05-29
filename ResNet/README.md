## ResNet

This repository contains two files, `res.ipynb` and `resNew.ipynb`, both inspired from [this paper.](https://arxiv.org/pdf/1512.03385.pdf)

#### res.ipynb
This was the initial attempt at coding the network. It uses an OOP approach which was motivated by the [tensorflow documentation.](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
It has:
- Two layer classes:
  - `resId`: Used for an identity shortcut in which dimensions of both the input and the residual is kept the same by padding with zeroes.
  - `resSh`: Used for a shortcut in which a `1x1 convolution` is applied on the input before adding with the residual to make their dimensions the same.
- One model class which utilises the classes above to make a model based on resNet.
It was only able to reach **53%** accuracy on test images.

#### resNew.ipynb
This contains a lot of modifications/ improvements immplemented on `res.ipynb`, with [this implementation](https://keras.io/zh/examples/cifar10_resnet/) being used as reference.
For instance, the use of kernel regularizers, kernel initializers, `categorical_crossentropy` as loss, data augmentation, and subtracting mean of training data from testing data.
Increased accuracy to **66%** on test images, which is an improvement over `res.ipynb`. 
(PS. I accidentally printed the whole loss and accuracy log of traning dataset instead of printing the last log, so, for convenience, last loss was `0.9141` and last accuracy was **75.43%**.
