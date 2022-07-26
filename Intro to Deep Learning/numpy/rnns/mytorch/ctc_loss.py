import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """

        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        # <---------------------------------------------
        ctc = CTC()
        self.gamma_bag = []
        
        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            #import pdb
            #pdb.set_trace()
            
            batch_target = self.target[b, 0:self.target_lengths[b]]
            batch_target, batch_skip_connections = ctc.targetWithBlank(batch_target)
            batch_target = batch_target.astype(int)
            batch_logits = self.logits[0:self.input_lengths[b], b, :]
            
            alpha_prob_batch = ctc.forwardProb(batch_logits, batch_target, batch_skip_connections)
            beta_prob_batch = ctc.backwardProb(batch_logits, batch_target, batch_skip_connections)
            
            batch_gamma = ctc.postProb(alpha_prob_batch, beta_prob_batch)
            

            for time_step in range(self.input_lengths[b]):
                for sequence_index in range(len(batch_target)):
                    totalLoss[b] += batch_gamma[time_step, sequence_index] * np.log(batch_logits[time_step, batch_target[sequence_index]]) 
            
            #Used in the backwards function 
            self.gamma_bag.append(batch_gamma)
        return -(totalLoss.sum()/B)

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: (seqlength, batch_size, len(Symbols))
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------
        ctc = CTC()

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------
            batch_target = self.target[b, 0:self.target_lengths[b]]
            batch_logits = self.logits[0:self.input_lengths[b], b, :]
            batch_target, batch_skip_connections = ctc.targetWithBlank(batch_target)
            batch_target = batch_target.astype(int)
            gamma = self.gamma_bag[b]
            
            for time_step in range(self.input_lengths[b]):
                for sequence_index in range(len(batch_target)):
                    dY[time_step, b, batch_target[sequence_index]] -= gamma[time_step, sequence_index]/batch_logits[time_step,batch_target[sequence_index]]

        return dY
