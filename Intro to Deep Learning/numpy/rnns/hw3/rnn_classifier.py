import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = [
             RNNCell(input_size, hidden_size)
             if i == 0
             else RNNCell(hidden_size, hidden_size)
             for i in range(num_layers)
         ]
        
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []
        

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [[W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                    [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1], ...]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(*linear_weights)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size)

        Output: logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if h_0 is None:
            hidden = np.zeros(
                (self.num_layers, batch_size, self.hidden_size), dtype=float
            )
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())

        ### Add your code here --->
        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
        #   Iterate over the length of your self.rnn (through the layers)
        #       Run the rnn cell with the correct parameters and update
        #       the parameters as needed. Update hidden.
        #   Similar to above, append a copy of the current hidden array to the hiddens list

        # Get the outputs from the last time step using the linear layer and return it
        # logits =
        # <--------------------------
        
        for time_index in range(seq_len):
            
            #We must store our hidden layers for use throughout the network
            hidden_states_current_time_all_layers = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype=float)
            
            for layer_index in range(self.num_layers):
                
                #We must first feed input data into network
                if layer_index == 0 :
                    hidden_state_current_time = self.rnn[layer_index](x[:,time_index, :],self.hiddens[time_index][layer_index])
                    hidden_states_current_time_all_layers[layer_index, :, :] = hidden_state_current_time

                else:
                    hidden_state_current_time= self.rnn[layer_index](hidden_state_current_time,self.hiddens[time_index][layer_index])
                    hidden_states_current_time_all_layers[layer_index, :, :] = hidden_state_current_time
            
            #After we have all the hidden states from the timestep, let's store them for future use
            self.hiddens.append(hidden_states_current_time_all_layers.copy())
        

        #After we made it through all layers and time steps, we must take the last hidden layer and feed it through a linear layer
        last_hidden_state = self.hiddens[-1][-1, :, :]
        linear_output = self.output_layer(last_hidden_state)

        return(linear_output)

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)
                

        """

        * Notes:
        * More specific pseudocode may exist in lecture slides and a visualization
          exists in the writeup.
        * WATCH out for off by 1 errors due to implementation decisions.

        Pseudocode:
        * Iterate in reverse order of time (from seq_len-1 to 0)
            * Iterate in reverse order of layers (from num_layers-1 to 0)
                * Get h_prev_l either from hiddens or x depending on the layer
                    (Recall that hiddens has an extra initial hidden state)
                * Use dh and hiddens to get the other parameters for the backward method
                    (Recall that hiddens has an extra initial hidden state)
                * Update dh with the new dh from the backward pass of the rnn cell
                * If you aren't at the first layer, you will want to add
                  dx to the gradient from l-1th layer.

        * Normalize dh by batch_size since initial hidden states are also treated
          as parameters of the network (divide by batch size)

        """


        for time_index in reversed(range(seq_len)):                       
            for layer_index in reversed(range(self.num_layers)):

                if layer_index == 0:
                    res_dx, dh[layer_index]= self.rnn[layer_index].backward(dh[layer_index], self.hiddens[time_index+1][layer_index],self.x[:, time_index,:],self.hiddens[time_index][layer_index])
                
                else: 
                    res_dx, dh[layer_index]= self.rnn[layer_index].backward(dh[layer_index], self.hiddens[time_index+1][layer_index],self.hiddens[time_index+1][layer_index-1],self.hiddens[time_index][layer_index])
                    dh[layer_index-1] = (res_dx + dh[layer_index-1])

        return dh/batch_size

