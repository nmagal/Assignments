"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import BatchNorm
from linear import Linear


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initializingr linear layers into the list 'self.linear_layers'
        self.linear_layers = []
        
        #If we have a mlp we will have a different init then linear regression
        if len(hiddens) > 0:
            first_layer = Linear(input_size,hiddens[0],weight_init_fn,bias_init_fn)
            self.linear_layers.append(first_layer)
            for layer_size_index in range(len(hiddens)-1):
                self.linear_layers.append(Linear(hiddens[layer_size_index],hiddens[layer_size_index+1],weight_init_fn, bias_init_fn))
            last_layer = Linear(hiddens[-1], output_size, weight_init_fn, bias_init_fn)
            self.linear_layers.append(last_layer)
        
        if len(hiddens) == 0:
            self.linear_layers.append(Linear(input_size, output_size, weight_init_fn, bias_init_fn))
        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = []
            for batch_layer in range(num_bn_layers):
                new_batch_norm = BatchNorm(hiddens[batch_layer])
                self.bn_layers.append(new_batch_norm)
               
                
    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Forward pass through MLP
        for index in range(len(self.linear_layers)):
            x = self.linear_layers[index](x)
            if self.bn and (index < len(self.bn_layers)):
                x = self.bn_layers[index](x,self.train_mode)
            x = self.activations[index](x)
        return(x)


    def zero_grads(self):
        for layer in self.linear_layers:
            layer.dW.fill(0.0)
            layer.db.fill(0.0)
        
        #If we are using batch norm we must also do this
        if self.bn:
        
            for batch_norm_layer in self.bn_layers:
                batch_norm_layer.dgamma.fill(0.0)
                batch_norm_layer.dbeta.fill(0.0)
                

    def step(self):
        # Applying a step to the weights and biases of the linear layers and batchnorm layers
        for hidden_layer in self.linear_layers:
            hidden_layer.W = hidden_layer.W + hidden_layer.momentum_W
            hidden_layer.b = hidden_layer.b + hidden_layer.momentum_b
            
        if self.bn:
            for batch_norm_layer in self.bn_layers:
                batch_norm_layer.gamma = batch_norm_layer.gamma - (self.lr*batch_norm_layer.dgamma)
                batch_norm_layer.beta = batch_norm_layer.beta - (self.lr*batch_norm_layer.dbeta)


    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and linear layers
        
        loss = self.criterion(self.linear_layers[-1].forward_value,labels)
        grad_loss_wr_Ylin = self.criterion.derivative()
        

        for index in reversed(range(len(self.linear_layers))):
              grad_loss_wr_Y = self.linear_layers[index].backward(grad_loss_wr_Ylin)
              
              #Updating Momentum
              self.linear_layers[index].momentum_W = self.linear_layers[index].momentum_W * self.momentum - self.lr * self.linear_layers[index].dW
              self.linear_layers[index].momentum_b = self.linear_layers[index].momentum_b * self.momentum - self.lr * self.linear_layers[index].db
              
              if index!= 0:
                  grad_loss_wr_Ylin = self.activations[index-1].derivative()*grad_loss_wr_Y
              
              if self.bn and (index <= len(self.bn_layers)) and index!=0:
                  grad_loss_wr_Ylin = self.bn_layers[index-1].backward(grad_loss_wr_Ylin)


    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()


    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()


    def __call__(self, x):
        return self.forward(x)


    def train(self):
        self.train_mode = True


    def eval(self):
        self.train_mode = False
