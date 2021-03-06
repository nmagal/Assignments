# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *

class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        # <---------------------
        
        self.conv1 = Conv1D(in_channel = num_input_channels, out_channel = 56, kernel_size=5, stride = 1, weight_init_fn =conv_weight_init_fn, bias_init_fn = bias_init_fn)
        self.conv2 = Conv1D(in_channel = 56, out_channel = 28, kernel_size=6, stride = 2, weight_init_fn =conv_weight_init_fn, bias_init_fn = bias_init_fn)
        self.conv3 = Conv1D(in_channel = 28, out_channel = 14, kernel_size=2, stride = 2, weight_init_fn =conv_weight_init_fn, bias_init_fn = bias_init_fn)
        self.flatten = Flatten()
        
        self.convolutional_layers = [self.conv1,self.conv2,self.conv3]

        self.linear_layer = Linear(420, num_linear_neurons,linear_weight_init_fn,bias_init_fn )


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """

        ## Your code goes here -->
        # Iterate through each layer
        # <---------------------
        
        # Save output (necessary for error and loss)
        for layer in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[layer](x)
            x = self.activations[layer](x)
        
        x = self.flatten(x)
        x = self.linear_layer(x)

        
        self.output = x

        return self.output

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """
        print("hello there part 2!")

        m, _ = labels.shape
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()

        ## Your code goes here -->
        # Iterate through each layer in reverse order
        # <---------------------
        grad = self.linear_layer.backward(grad)
        grad = self.flatten.backward(grad)
        
        for layer_index in reversed(range(len(self.convolutional_layers))):
            grad = self.activations[layer_index].derivative()*grad
            grad = self.convolutional_layers[layer_index].backward(grad)
            
        
        
        return grad


    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
