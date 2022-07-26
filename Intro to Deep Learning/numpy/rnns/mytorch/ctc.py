import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]
        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]

        """
        extSymbols = []
        skipConnect = []

        # -------------------------------------------->

        # Your Code goes here
        extSymbols = np.zeros((2* len(target)))
        extSymbols[::2] = target
        extSymbols = np.insert(extSymbols, self.BLANK, 0)
        
        skipConnect = np.zeros((len(extSymbols))).astype(int)
        for index in reversed(range(len(skipConnect))):
            if extSymbols[index] == self.BLANK or extSymbols[index] == extSymbols[index-2] or index==0 or index==1:
                skipConnect[index] = 0
            else:
                skipConnect[index] = 1
        
        return extSymbols, skipConnect

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))
        extSymbols = extSymbols.astype(int)

        alpha[0,0] = logits[0, extSymbols[0]]
        alpha[0,1] = logits[0, extSymbols[1]]
        alpha[0, 2:] = 0
        
        for t in range(T-1):
            alpha[t+1, 0] = alpha[t, 0] * logits[t+1, extSymbols[0]]
            for alpha_index in range(S-1):
                alpha[t+1, alpha_index+1] = alpha[t, alpha_index] + alpha[t, alpha_index+1]
                if skipConnect[alpha_index+1]:
                    alpha[t+1, alpha_index+1] += alpha[t, alpha_index-1]
                alpha[t+1, alpha_index+1] *= logits[t+1, extSymbols[alpha_index+1]]        
               
        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))
        extSymbols = extSymbols.astype(int)
        
        beta[T-1, -1] = 1
        beta[T-1, -2] = 1
        beta[T-1, :-2] = 0
        
        for t in reversed(range(T-1)):
            beta[t, -1] = beta[t+1, -1]*logits[t+1, extSymbols[-1]]
            for beta_index in reversed(range(S-1)):
                beta[t, beta_index] = beta[t+1, beta_index] * logits[t+1, extSymbols[beta_index]] + beta[t+1, beta_index+1]* logits[t+1, extSymbols[beta_index+1]]
                if (beta_index < S-2 and skipConnect[beta_index+2]):
                    beta[t, beta_index] += beta[t+1, beta_index+2] *logits[t+1, extSymbols[beta_index+2]]
            
        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros(T)

        for t in range(T):
            for i in range(S):
                gamma[t, i] = alpha[t, i] * beta[t,i]
                sumgamma[t] += gamma[t,i]
            gamma[t] /= gamma[t].sum()
                
        return gamma
