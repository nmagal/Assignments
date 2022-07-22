#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:17:35 2021

@author: nicholasmagal
"""
import numpy as np
import activation
import linear
import batchnorm
import loss
import tester


'''
#Testing the linear layer on a single input
X = np.array([[4,3]])
W = np.array([[4, 2, -2],[5, 4, 5]])
B = np.array([[1, 2, 3]])
linearr = linear.Linear(None, None, None,None, W,B)
Y1lin = linearr.forward(X)

#Testing the Sigmoid activation
sigmoid = activation.Sigmoid()
Y1 = sigmoid.forward(Y1lin) 

#Testing the ReLU Activation
reLU = activation.ReLU()
Y1 = reLU.forward(Y1lin)

#Testing the Tanh
tanh = activation.Tanh()
Y1 = tanh.forward(Y1lin)

#Testing Softmax
softmax_cross_entropy = loss.SoftmaxCrossEntropy()
Y1 = softmax_cross_entropy.softmax(Y1lin)

#Testing the MLP on a single input
X = np.array([[4,3]])
W1 = np.array([[ 4, 2, -2],[ 5, 4, 5]])
B1 = np.array([[ 0, 0, 0]])
W2 = np.array([[ 2, 5, 6],[ 2, -3, -3],[-2, 4, 3]])
B2 = np.array([[ 0, 0, 0]])
W3 = np.array([[ 5, 5],[ 3, -1],[ 5, 4]])
B3 = np.array([[ 0, 0]])

hiddenlayer = linear.Linear(None, None, None,None, W1,B1)
Y1lin = hiddenlayer.forward(X)
identity_activation = activation.Identity()
Y1 = identity_activation(Y1lin)

hiddenlayer_2 = linear.Linear(None,None,None,None, W2, B2)
Y2lin = hiddenlayer_2(Y1)
Y2 = identity_activation(Y2lin)

hiddenlayer3 = linear.Linear(None,None,None,None, W3, B3)
Y3 = hiddenlayer3(Y2)
Y3 = identity_activation(Y3)

#Implementing for a batch of inputs
X = np.array([[4,3], [5,6], [7,8]])
W1 = np.array([[ 4, 2, -2],[ 5, 4, 5]])
B = np.array([[1,2,3]])

Y1lin = linearr.forward(X)

#Implementing an activation layer Sigmoid
Y1 = sigmoid.forward(Y1lin)
Y1 = reLU.forward(Y1lin)
Y1 = tanh.forward(Y1lin)
Y1 = softmax_cross_entropy.softmax(Y1lin)

#Implementing a complete MLP
X = np.array([[4,3],[5,6],[7,8]])
W1 = np.array([[ 4, 2, -2],[ 5, 4, 5]])
B1 = np.array([[ 0, 0, 0]])
W2= np.array([[ 2, 5, 6],[ 2, -3, -3],[-2, 4, 3]] )
B2 = np.array([[ 0, 0, 0]])
W3 = np.array([[ 5, 5],[ 3, -1],[ 5, 4]] )
B3 = np.array([[ 0, 0]])

hidden_layer_1 = linear.Linear(None,None,None,None, W1, B1)
Y1lin = hidden_layer_1(X)

hidden_layer_2 = linear.Linear(None,None,None,None, W2,B2)
Y2lin = hidden_layer_2(Y1lin)

hidden_layer_3 =linear.Linear(None,None,None,None, W3,B3)
Y3lin = hidden_layer_3(Y2lin)

#Computing the loss
Y= np.array([[1,2,3]])
D = np.array([[2,3,4]])
l2_loss = loss.L2Loss()
loss = l2_loss.forward(Y,D)
D = np.array([[1540,900]])
loss = l2_loss.forward(Y3,D)

#Softmax check
Y = np.array([[0.2,0.3,0.5]])
c = np.array([[2]])
loss = softmax_cross_entropy.forward(Y,c)
c = np.array([[1]])
loss = softmax_cross_entropy.forward(Y3,c)

#Testing the gradients 
Y = np.array([[1, 2, 3]])
D = np.array([[2, 3, 4]])
out = l2_loss.forward(Y,D)
backwards = l2_loss.derivative()

D = np.array([[240, 4]])
out = l2_loss.forward(Y3,D)
backwards = l2_loss.derivative()

#testing gradient for softmax.... Potential issue here, softmax is not quite the same 
Y = np.array([[0.2,0.3,0.5]])
c = np.array([[1]])
Y3_softmax = softmax_cross_entropy.softmax(Y3)
out = softmax_cross_entropy.forward(Y3_softmax,c)
backwards = softmax_cross_entropy.derivative()

#%% This is one hot encoder machine 
        #one_hot_classes = np.zeros((row_dim,col_dim))
               
        #for batch_index in range(row_dim):
            #one_hot_classes[batch_index][y[batch_index][0]-1] = 1

z = np.array([[4,3]])
W = np.array([[4, 2, -2], [5, 4, 5]])
B = np.array([[1,1,1]])
gradY = np.array([[1,-1,1]])
linear_layer = linear.Linear(2,3, None, None)
out = linear_layer.forward(z)
backwards = linear_layer.backward(gradY)

#Section 3.5 
X = np.array([[4,3],[5,6],[7,8]])
W1 = np.array([[ 4, 2, -2],[ 5, 4, 5]])
B1 = np.array([[ 0, 0, 0]])
W2= np.array([[ 2, 5, 6],[ 2, -3, -3],[-2, 4, 3]] )
B2 = np.array([[ 0, 0, 0]])
W3 = np.array([[ 5, 5],[ 3, -1],[ 5, 4]] )
B3 = np.array([[ 0, 0]])

hidden_layer_1 = linear.Linear(None,None,None,None, W1, B1)
Y1lin = hidden_layer_1(X)
reLu = activation.ReLU()
Y1 = reLu.forward (Y1lin)

hidden_layer_2 = linear.Linear(None,None,None,None, W2,B2)
Y2lin = hidden_layer_2(Y1lin)
reLu_2 = activation.ReLU()
Y2 = reLu_2.forward(Y2lin)

hidden_layer_3 =linear.Linear(None,None,None,None, W3,B3)
Y3lin = hidden_layer_3(Y2lin)
reLu_3 = activation.ReLU()

D = np.array([[240,4],[320,2],[69,5]])

#Backwards Pass
loss_function = loss.SoftmaxCrossEntropy()
loss_forward = loss_function.softmax(Y3lin) 
gradY3lin = loss_function.softmax_debug_backwards(loss_forward,D )

gradY2 = hidden_layer_3.backward(gradY3lin)

gradY2lin = reLu_2.derivative() * gradY2
gradY1 = hidden_layer_2.backward(gradY2lin)

gradY1lin = reLu.derivative() * gradY1
gradX = hidden_layer_1.backward(gradY1lin)

def SGD(lr, parameter_to_update, parameter_gradient):
    parameter_to_update = parameter_to_update - lr*(parameter_gradient)
    return parameter_to_update 

#Updating weights 
updated_W1 = SGD(.1, W1, hidden_layer_1.dW)
updated_b1 = SGD(.1, B1, hidden_layer_1.db)


input_size = 784
output_size = 10
hiddens = [64,32]
activations = [activation.Sigmoid(),activation.Sigmoid(), activation.Identity()]



#%%
# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
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

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
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
        # Complete the forward pass through your entire MLP.
        output = np.zeros(None)
        for index in range(len(self.linear_layers)):
            x = self.linear_layers[index](x)
            if self.bn and (index < len(self.bn_layers)):
                x = self.bn_layers[index](x)
            x = self.activations[index](x)
        return(x)

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        raise NotImplemented

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            pass
        # Do the same for batchnorm layers

        raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        
        #This only works for the simple case 
        #loss = self.criterion(self.linear_layers[-1].forward_value,labels)
        #dr_loss_wr_Y3lin = self.criterion.derivative()
        #gradY2 = self.linear_layers[-1].backward(dr_loss_wr_Y3lin) 
        
        loss = self.criterion(self.linear_layers[-1].forward_value,labels)
        grad_loss_wr_Ylin = self.criterion.derivative()
        

        for index in reversed(range(len(self.linear_layers))):
              grad_loss_wr_Y = self.linear_layers[index].backward(grad_loss_wr_Ylin)
              
              grad_loss_wr_Ylin = self.activations[index-1].derivative()*grad_loss_wr_Y
              
              if self.bn and (index < len(self.bn_layers)):
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

#This function does not carry any points. You can try and complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented

#labels= np.array([[5, 13], [5,13], [5,13]])
#runner = MLP(input_size, output_size, hiddens, activations,linear.weight_matrix_init, linear.weight_bias_init,loss.SoftmaxCrossEntropy(),None,None, num_bn_layers=1)
#input_x = np.ones((1,784))
#cat = runner.forward(input_x)
#dog = runner.backward(labels)





#Testing forward for batch norm
#X = np.array([[4,3],[5,6],[7,8]])
W1 = np.array([[ 4, 2, -2],[ 5, 4, 5]])
B1 = np.array([[ 0, 0, 0]])
W2= np.array([[ 2, 5, 6],[ 2, -3, -3],[-2, 4, 3]] )
B2 = np.array([[ 0, 0, 0]])
W3 = np.array([[ 5, 5],[ 3, -1],[ 5, 4]] )
B3 = np.array([[ 0, 0]])

hidden_layer_1 = linear.Linear(None,None, W1, B1)
forward_1_nuerons = hidden_layer_1.forward(X)

batch_norm_1 = batchnorm.BatchNorm(forward_1_nuerons.shape[1])
norm_1_nuerons = batch_norm_1(forward_1_nuerons)

reLu_1 = activation.ReLU()
reLu_1_nuerons = reLu_1(norm_1_nuerons)

hidden_layer_2 = linear.Linear(None,None, W2, B2)
hidden_layer_2_nuerons = hidden_layer_2(reLu_1_nuerons)

batch_norm_2 = batchnorm.BatchNorm(hidden_layer_2_nuerons.shape[1])
batch_norm_2_nuerons = batch_norm_2(hidden_layer_2_nuerons)

reLU_2 = activation.ReLU()
reLU_2_nuerons = reLU_2(batch_norm_2_nuerons)

hidden_layer_3 = linear.Linear(None,None, W3, B3)
hidden_layer_3_outputs = hidden_layer_3(reLU_2_nuerons)

loss_function = loss.SoftmaxCrossEntropy()
labels= np.array([[5, 13], [5,13], [5,13]])
softmax_output= loss_function.softmax(hidden_layer_3_outputs)

#Backprop
gradY3lin = loss_function.softmax_debug_backwards(softmax_output,labels)
gradY2 = hidden_layer_3.backward(gradY3lin)

grad_relu_Y2 = reLU_2.derivative() * gradY2

grad_bn_Y2 = batch_norm_2.backward(grad_relu_Y2)

mlp=tester.MLP(784, 10, [64, 32], [activation.Sigmoid(), activation.Sigmoid(), activation.Identity()],
                      linear.weight_init, linear.bias_init, loss.SoftmaxCrossEntropy(), 0.008,
                      momentum=0.0, num_bn_layers=1)
input_x = np.ones((1,784))
mlp.forward(input_x)
#labels = np.array([[0,0,0,0,0,0,0,0,0,1]])
#mlp.backward(labels)

X = np.array([[10,14]])
W1 = np.array([[ 4, 2, -2]])
B1 = np.array([[ 0, 0, 0]])
hidden_layer_2 = linear.Linear(None,None, W1, B1)
cat = hidden_layer_2(X)
'''
labels = np.array([[0,0,0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,0,0,0,0,1]])

input_x = np.array([[3,4,3,5,6,7,8,9,6,5],
                    [3,4,3,5,6,7,8,9,6,5]])
loss = loss.SoftmaxCrossEntropy()
cat = loss.forward(input_x,labels)
