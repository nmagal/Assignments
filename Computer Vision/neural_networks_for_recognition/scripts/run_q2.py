import numpy as np
from nn import *
from util import *

# fake data in form N x D 
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])

# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters dictionary 
params = {}

# Q 2.1
# initialize network
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))

# Q 2.4 create batches
batches = get_random_batches(x,y,5)
batch_num = len(batches)

# Training loop 
max_iters = 500
learning_rate = 1e-3
for itr in range(max_iters):
    
    #Calculating loss/acc over entire dataset
    h1 = forward(x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, acc = compute_loss_and_acc(y, probs)
    if itr % 100 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,loss,acc))
    
    for xb,yb in batches:
        
        # forward pass
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # backward pass
        delta1 = probs - yb
        dz1 = backwards(delta1, params, 'output', activation_deriv=linear_deriv)
        dx = backwards(dz1, params, 'layer1', activation_deriv=sigmoid_deriv)
        
        # Mini batch SGD step  
        params['Woutput'] = params['Woutput'] - (learning_rate*params['grad_Woutput'].T)
        params['boutput'] = params['boutput'] - (learning_rate*np.sum(params['grad_boutput']))
        
        params['Wlayer1'] = params['Wlayer1'] - (learning_rate*params['grad_Wlayer1'].T)
        params['blayer1'] = params['blayer1'] - (learning_rate*np.sum(params['grad_blayer1']))

# Store gradients from backprop for comparision with finite difference method 
h1 = forward(x,params,'layer1')
probs = forward(h1,params,'output',softmax)
loss, acc = compute_loss_and_acc(y, probs)
delta1 = probs - y
delta2 = backwards(delta1,params,'output',linear_deriv)
backwards(delta2,params,'layer1',sigmoid_deriv)

# save the old params
import copy
params_orig = copy.deepcopy(params)

# compute gradients using finite difference
eps = 1e-6
for k,v in params.items():
    if '_' in k: 
        continue
    # we have a real parameter!
    # for each value inside the parameter
    #   add epsilon
    #   run the network
    #   get the loss
    #   subtract 2*epsilon
    #   run the network
    #   get the loss
    #   restore the original parameter value
    #   compute derivative with central diffs
    
    #This is for calcualting the gradient for the weight
    if 'W' in k:
        for row_index in range(v.shape[0]):
            for col_index in range(v.shape[1]):
                #adding slight perturbation
                og_value = params[k][row_index, col_index]
                params[k][row_index, col_index] = params[k][row_index, col_index] + eps
                
                #running network
                h1 = forward(x,params,'layer1')
                probs = forward(h1,params,'output',softmax)
                loss, acc = compute_loss_and_acc(y, probs)
                
                #sub slight perturbation
                params[k][row_index, col_index] = params[k][row_index, col_index] - 2*eps
                
                #running network
                h1 = forward(x,params,'layer1')
                probs = forward(h1,params,'output',softmax)
                loss2, acc = compute_loss_and_acc(y, probs)
                
                #restoring og value
                params[k][row_index, col_index] = og_value
                
                #Computing derivative
                if 'output' in k:
                    params['grad_Woutput'][col_index][row_index]= (loss - loss2)/(2*eps)
                else:
                    params['grad_Wlayer1'][col_index][row_index] = (loss - loss2)/(2*eps)
                    
    #This is for calculating the gradient for the bias
    if 'b' in k:
        #adding slight perturbation
        og_value = params[k]
        params[k] = params[k] + eps
        
        #running network
        h1 = forward(x,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        loss, acc = compute_loss_and_acc(y, probs)
                
        #sub slight perturbation
        params[k]= params[k] - 2*eps
        
        h1 = forward(x,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        loss2, acc = compute_loss_and_acc(y, probs)
        
        #restoring og value
        params[k]= og_value
        
        #Computing derivative
        if 'output' in k:
            params['grad_boutput'] = (loss - loss2)/(2*eps)
        else:
            params['grad_blayer1'] = (loss - loss2)/(2*eps)

#Comparing finite gradient difference vs backprop
total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('total {:.2e}'.format(total_error))
