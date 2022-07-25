# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
    

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        #numerical stability trick 
        exps = np.exp(x - np.max(x, axis=1)[:,None])
        self.sm = exps / np.sum(exps, axis=1)[:,None]
        loss = -np.log((self.sm*y).sum(axis=1))
        return loss

    def derivative(self):
        return self.sm - self.labels

class L2Loss(Criterion):
    
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        
        row_dim, col_dim = x.shape
        loss = 0
        loss_per_batch = []
        
        for row_index in range(row_dim):
            for col_index in range(col_dim):
                loss = (x[row_index][col_index] - y[row_index][col_index])**2 + loss
 
            loss_per_batch.append(loss/col_dim)
        
        self.loss = np.array(loss_per_batch).reshape(-1,1)
        return self.loss


    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        return(self.logits - self.labels)

        