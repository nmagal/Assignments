import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string
from util import *

#Running an autoencoder 

#Get training and validation data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x = train_data['train_data']
valid_x = valid_data['valid_data']

#Hyperparameters 
max_iters = 100
batch_size = 36 
learning_rate =  3e-6 # 5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers
initialize_weights(train_x.shape[1], 32, params,'layer1')
initialize_weights(32, 32, params,'layer2')
initialize_weights(32, 32, params,'layer3')
initialize_weights(32, 1024, params,'layer4')

# Training loop
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        
        #Forward pass
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'layer2', relu)
        h3 = forward(h2,params,'layer3', relu)
        reconstructed_image = forward(h3,params,'layer4')
        
        #loss
        total_loss += np.linalg.norm(xb-reconstructed_image)
        
        #Backwards pass
        delta1 = 2*(reconstructed_image - xb)
        dz4 = backwards(delta1, params, 'layer4', activation_deriv=linear_deriv)
        dz3 = backwards(dz4, params, 'layer3', activation_deriv=relu_deriv)
        dz2 = backwards(dz3, params, 'layer2', activation_deriv=relu_deriv)
        dz1 = backwards(dz2, params, 'layer1', activation_deriv=relu_deriv)
            
        #Updating momentum
        for i in range(1,5):
            params['m_Wlayer' + str(i)] = (params['m_Wlayer' + str(i)]*.9) - (learning_rate*params['grad_Wlayer' + str(i)].T)
            params['m_blayer' +str(i)] = params['m_blayer' +str(i)]*.9 - (learning_rate*np.sum(params['grad_blayer'+str(i)]))
            
            
        #Updating weights
        for i in range(1,5):
            params['Wlayer' + str(i)] = params['Wlayer' + str(i)] + params['m_Wlayer' + str(i)]
            params['blayer'+ str(i)] = params['blayer'+ str(i)] + np.sum(params['m_blayer'+ str(i)])
       
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.10f}".format(itr,total_loss/train_x.shape[0]))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
# both will be plotted below
##########################
##### your code here #####
##########################
h1 = forward(visualize_x,params,'layer1',relu)
h2 = forward(h1,params,'layer2', relu)
h3 = forward(h2,params,'layer3', relu)
reconstructed_x = forward(h3,params,'layer4')

# plot visualize_x and reconstructed_x
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
PSNR=0
for image in valid_x:
    #Running our image through the network
    h1 = forward(image,params,'layer1',relu)
    h2 = forward(h1,params,'layer2', relu)
    h3 = forward(h2,params,'layer3', relu)
    reconstructed_x = forward(h3,params,'layer4')
    
    #Getting total PSNR 
    PSNR += peak_signal_noise_ratio(image, reconstructed_x)
print("Total PSNR is: ", str(PSNR/valid_x.shape[0]))
    
