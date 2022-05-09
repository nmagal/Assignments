import scipy.io
import warnings
import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):

    #Init with Xavier init
    min_weight_value = -np.sqrt(6)/(np.sqrt(in_size + out_size))
    max_weight_value = np.sqrt(6)/(np.sqrt(in_size + out_size))
    
    #Creating weight matrix and bias, weight matrix is ouput_size x input_size (must transpose input layer for WX)
    W = np.random.uniform(min_weight_value,max_weight_value, (out_size,in_size)).T
    b = np.random.uniform(min_weight_value,max_weight_value, (out_size))
    ##########################
    
    #Adding to our dictionary
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# a sigmoid activation function
def sigmoid(x):

    res = 1/(1+np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    
    # get the layer parameters
    W = params['W' + name]# params['Wlayer1']
    b = params['b' + name]# params['blayer1']

    #Getting pre and post activation of neurons 
    pre_act = (W.T@X.T).T + b
    post_act = activation(pre_act)

    # storing the pre-activation and post-activation values for backprop later
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax function 
def softmax(x):

    #Numeric Stability Trick 
    c = np.amax(x, axis =1, keepdims=True)
    x = x-c
    x_exp = np.exp(x)
    x_sums = np.sum(x_exp, axis=1, keepdims = True)
    softmax = x_exp/x_sums

    return softmax

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    
    #Calculating loss
    one_hot_probs = np.sum(probs * y, axis = 1, keepdims = True)
    loss = -np.sum(np.log(one_hot_probs))
    
    #Calculating acc
    predictions = np.argmax(probs, axis = 1)
    ground_truth = np.argmax(y,axis=1)
    acc = (np.sum(predictions == ground_truth))/y.shape[0]
    
    return loss, acc 

#Sigmoid derivative
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

    
def backwards(delta,params, name='', activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass
    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # Reading in values
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # doing the derivative through activation first
    grad_h = activation_deriv(post_act)
    
    #doing the derivative for the hidden layer (before activation)
    grad_h = delta * grad_h 
    
    #Getting the derivative of x, w and b
    grad_X = grad_h @ W.T
    grad_W = grad_h.T @ X
    grad_b = (np.sum(grad_h))
    
    # store the gradients
    params['grad_W' + name] = grad_W#/X.shape[0]
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):

    ##########################
    ##### your code here #####
    x = [x[i*batch_size:(i+1)*batch_size] for i in range(x.shape[0]//batch_size)]
    y = [y[i*batch_size:(i+1)*batch_size] for i in range(y.shape[0]//batch_size)]
    batches = list(zip(x,y)) 
    np.random.shuffle(batches)   
    ##########################
    return batches

