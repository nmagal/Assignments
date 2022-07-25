# Introduction to Deep Learning 

Below contains coding assignment descriptions for Intro to Deep Learning. The coding assignments can be divided into two components, numpy implementations of a deep learning architectures and pytorch implementations of deep learning architectures.

## Pytorch Implementations 

### Neural Network Challenge
Kaggle Challenge: [Frame level classification of speech](https://www.kaggle.com/competitions/11-785-s20-hw1p2/overview). In this challenge, a neural network is created for frame level classification of speech. Data is given in the form of raw mel spectrogram frames, and our predictions are phoneme state labels.

### Resnet Classification and Verification Challenge
Kaggle Challenge: [Face Classification](https://www.kaggle.com/c/11-785-s20-hw2p2-classification) and [Face Verification](https://www.kaggle.com/c/11-785-s20-hw2p2-verification).  In this challenge, Resnet34 is created for the task of facial classification and verification. Resnet50 is also implemented, but is not used due to difficulties in training stability. By using embeddings created by Resnet34 trained on classification data, we are able to compute image pair distances in order to perform Face Verification. 


## Numpy Implementations 

### Neural Network
In this assignment, a neural network is created. Linear layers, activation functions, backprop, batchnorm, and momentum are implemented.

### Convolutional Neural Network
In this assignment, a convolutional nueral network is created. This builds upon the previous assignment, and 1D and 2D convolutional layers are implemented.
