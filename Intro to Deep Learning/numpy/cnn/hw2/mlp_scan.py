# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        self.conv1 = Conv1D(in_channel=24, out_channel=8, kernel_size=8, stride=4)
        self.conv2 = Conv1D(in_channel=8, out_channel=16, kernel_size=1, stride=1)
        self.conv3 = Conv1D(in_channel=16, out_channel=4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1,w2,w3 = weights
        self.conv1.W = np.transpose((w1.T).reshape((8, 8, 24)), (0, 2, 1))
        self.conv2.W = np.transpose((w2.T).reshape((16, 1, 8)), (0, 2, 1))
        self.conv3.W = np.transpose((w3.T).reshape((4, 1, 16)), (0, 2, 1))

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1D(in_channel=24, out_channel=2, kernel_size=2, stride=2)
        self.conv2 = Conv1D(in_channel=2, out_channel=8, kernel_size=2, stride=2)
        self.conv3 = Conv1D(in_channel=8, out_channel=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.W = np.transpose((w1.T[0:2,0:48]).reshape((2, 2, 24)), (0, 2, 1))
        self.conv2.W = np.transpose((w2.T[0:8,0:4]).reshape((8, 2, 2)), (0, 2, 1))
        self.conv3.W = np.transpose((w3.T).reshape((4, 2, 8)), (0, 2, 1))

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
