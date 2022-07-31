# Introduction to Deep Learning 

Below contains coding assignment descriptions for Intro to Deep Learning. The coding assignments can be divided into two components, creating a custom pytorch like custom deep learning library with numpy and pytorch implementations of deep learning architectures used in Kaggle competitions.

## Pytorch Implementations 

### Neural Network Challenge
Kaggle Challenge: [Frame level classification of speech](https://www.kaggle.com/competitions/11-785-s20-hw1p2/overview). In this challenge, a neural network is created for frame level classification of speech. Data is given in the form of raw Mel spectrogram frames, and our predictions are phoneme state labels.

### Resnet Classification and Verification Challenge
Kaggle Challenge: [Face Classification](https://www.kaggle.com/c/11-785-s20-hw2p2-classification) and [Face Verification](https://www.kaggle.com/c/11-785-s20-hw2p2-verification).  In this challenge, Resnet34 is created for the task of facial classification and verification. Resnet50 is also implemented, but is not used due to difficulties in training stability. By using embeddings created by Resnet34 trained on classification data, we are able to compute image pair distances in order to perform Face Verification. 

### LSTMs/GRUs for Utterance to Phoneme Challenge
Kaggle Challenge: [Utterance to Phoneme Mapping](https://www.kaggle.com/c/11-785-s20-hw3p2). In this challenge we are tasked with predicting phonemes given mel spectrogram speech data. Unlike the first challenge, our labels are not explicitly aligned, and in order to deal with this, we use CTC Loss. Many different model designs are tested in this challenge, including using a Pyramidal LSTM and using Resnet blocks to embed speech data. An ablation report was created that can be read [here](https://github.com/nmagal/Assignments/blob/master/Intro%20to%20Deep%20Learning/pytorch/utterance_LSTMs/Ablation_Report.pdf).

### Text Generation 
In this assignment, we are challenged to create a language model to generate text. Our dataset is the WikiText-2 Language Dataset, and techniques from the paper [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf) are used.

### Sequence to Sequence Attention Model
In this assignment, simple dot product attention is implemented alongside an encoder decoder model. The goal of this assignment was to learn how attention worked, and our dataset was speech data with labeled text transcription. 


## Numpy Implementations 

### Neural Network
In this assignment, a neural network is created. Linear layers, activation functions, backprop, batch norm, and momentum are implemented.

### Convolutional Neural Network
In this assignment, a convolutional neural network is created. This builds upon the previous assignment, and 1D and 2D convolutional layers are implemented.

### RNNs

In this assignment, RNN and GRU libraries are created. Furthermore, CTC Loss as well as Beam Search are implemented. 
