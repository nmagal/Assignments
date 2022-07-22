# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """
        
        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)
        self.z = None 
        
        self.dW = np.zeros(None)
        self.db = np.zeros(None)

        self.momentum_W = np.zeros(None)
        self.momentum_b = np.zeros(None)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.z = x 
        affine_sum = np.dot(x, self.W) + self.b 
        self.forward_value = affine_sum
        return(affine_sum)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
             
        self.dW = np.dot(np.transpose(self.z),delta)
        #Averaging for batches
        self.dW = self.dW/self.z.shape[0]
        
        self.db = delta
        self.db = (np.sum(self.db, axis=0)/self.z.shape[0]).reshape(1,-1)
        
        dZ = np.dot(delta, np.transpose(self.W))
        return dZ


def weight_init(input_size, output_size):
    weight_matrix = np.random.random((input_size,output_size))
    return(weight_matrix)

def bias_init(output_size):
    bias_column = np.random.random((1, output_size))
    return bias_column

