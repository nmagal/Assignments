import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()
        

        # Define other variables to store forward results for backward here
        self.dr2 = np.zeros((1,h))
        self.dr3 = np.zeros((1,h))
        self.dz = np.zeros((1,h))
        self.dhidden = np.zeros((1,h))
        self.dr1 = np.zeros((1,h))
        self.dn = np.zeros((1,h))
        self.dz = np.zeros((1,h))
        self.dq6 = np.zeros((1,h))
        self.dq5 = np.zeros((1,h))
        self.dq3 = np.zeros((1,h))
        self.dq4 = np.zeros((1,h))
        self.dq2 = np.zeros((1,h))
        self.dr = np.zeros((1,h))
        self.dq1 = np.zeros((1,h))
        self.dy5 = np.zeros((1,h))
        self.dy4 = np.zeros((1,h))
        self.dy3 = np.zeros((1,h))
        self.dy2 = np.zeros((1,h))
        self.dy1 = np.zeros((1,h))
        self.dz5 = np.zeros((1,h))
        self.dz4 = np.zeros((1,h))
        self.dz3 = np.zeros((1,h))
        self.dz2 = np.zeros((1,h))
        self.dz1 = np.zeros((1,h))
        
        
    

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        #Solving in terms of unary and binary for ease of solving derivitives 
        #self.r = self.r_act(self.Wrx @ x + self.bir + self.Wrh @ h + self.bhr)
        self.z1 = self.Wrx @ x
        self.z2 = self.Wrh @ h
        self.z3 = self.z1 + self.z2
        self.z4 = self.z3 + self.bir
        self.z5 = self.z4 + self.bhr
        self.r = self.r_act(self.z5)
        
        #self.z = self.z_act(self.Wzx @ x + self.biz + self.Wzh @ h + self.bhz)
        self.y1 = self.Wzx @ x
        self.y2 = self.Wzh @ h
        self.y3 = self.y1 + self.y2
        self.y4 = self.y3 + self.biz
        self.y5 = self.y4 + self.bhz
        self.z = self.z_act(self.y5)

        #self.n = self.h_act(self.Wnx @ x + self.bin + self.r*(self.Wnh @ h + self.bhn))
        self.q1 = self.Wnx @ x
        self.q2 = self.q1 + self.bin
        self.q3 = self.Wnh @ h
        self.q4 = self.q3 + self.bhn
        self.q5 = self.q4 * self.r
        self.q6 = self.q5 + self.q2
        self.n = self.h_act(self.q6)
        
        #h_t = (1- self.z)* self.n + self.z * h
        self.r1 = 1 - self.z
        self.r2 = self.r1 * self.n
        self.r3 = self.z * h
        self.h_t = self.r2 + self.r3
        
        # Define your variables based on the writeup using the corresponding
        # names below.  
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        
        assert self.n.shape == (self.h,)
        assert self.h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        # return h_t
        return self.h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.
        #Keep in mind I had to squeeze some dimensions
        
                
        dx= np.zeros((1,self.d))
        
        dr2_res, dr3_res = self.deriv(delta, self.r2, self.r3, '+')
        self.dr2 += dr2_res
        self.dr3 += dr3_res

        dz_res, dh_res = self.deriv(self.dr3, self.z, self.hidden.reshape(-1,1), '*')
        self.dz += dz_res
        self.dhidden += dh_res
        
        dr1_res, dn_res = self.deriv(self.dr2, self.r1, self.n, '*')
        self.dr1 +=dr1_res
        self.dn +=dn_res
        
        _, dz_res = self.deriv(self.dr1, 0, self.z, '-')
        self.dz += dz_res
        
        #Starting on third equation
        dq6_res, _ = self.deriv(self.dn, self.q6, 0, 'tanh' )
        self.dq6 += dq6_res
        
        dq5_res, dq2_res = self.deriv(self.dq6, self.q5, self.q2, '+')
        self.dq5 += dq5_res
        self.dq2 += dq2_res
        
        dq4_res, dr_res = self.deriv(self.dq5, self.q4, self.r, '*')
        self.dq4 += dq4_res
        self.dr+= dr_res

        dq3_res, dbhn_res = self.deriv(self.dq4, self.q3, self.bhn, '+')
        self.dq3 += dq3_res
        self.dbhn += dbhn_res.squeeze()
        
        dWnh_res, dhidden_res = self.deriv(self.dq3, self.Wnh, self.hidden.reshape(-1,1), "@")
        self.dWnh += dWnh_res.T
        self.dhidden += dhidden_res
        
        dq1_res, dbin_res = self.deriv(self.dq2, self.q1, self.bin, "+")
        self.dq1 += dq1_res
        self.dbin += dbin_res.squeeze()
        
        dWnx_res, x_res = self.deriv(self.dq1, self.Wnx, self.x.reshape(-1,1), "@")
        self.dWnx += dWnx_res.T
        dx += x_res
        
        #Starting on the second equation
        dy5_res, _ = self.deriv(self.dz, self.y5, 0, "sigmoid")
        self.dy5+= dy5_res
        
        dy4_res, dbhz_res = self.deriv(self.dy5, self.y4, self.bhz, "+")
        self.dy4 += dy4_res
        self.dbhz += dbhz_res.squeeze()
        
        dy3_res, dbiz_res = self.deriv(self.dy4, self.y3, self.biz, "+")
        self.dy3 += dy3_res
        self.dbiz += dbiz_res.squeeze()
        
        dy1_res, dy2_res = self.deriv(self.dy3, self.y1, self.y2, "+")
        self.dy1+= dy1_res
        self.dy2 += dy2_res
        
        dWzh_res, dhidden_res = self.deriv(self.dy2, self.Wzh, self.hidden.reshape(-1,1), '@')
        self.dWzh += dWzh_res.T
        self.dhidden += dhidden_res
        
        dWzx_res, dx_res = self.deriv(self.dy1, self.Wzx, self.x.reshape(-1,1), '@')
        self.dWzx += dWzx_res.T
        dx+= dx_res
        
        #Starting on first equation
        dz5_res, _ = self.deriv(self.dr, self.z5, 0, "sigmoid")
        self.dz5 += dz5_res
        
        dz4_res, dbhr_res = self.deriv(self.dz5, self.z4, self.bhr, '+')
        self.dz4 += dz4_res
        self.dbhr += dbhr_res.squeeze()
        
        dz3_res, dbir_res = self.deriv(self.dz4, self.z3, self.bir, "+")
        self.dz3 += dz3_res
        self.dbir += dbir_res.squeeze()
        
        dz1_res, dz2_res = self.deriv(self.dz3, self.z1, self.z2, "+")
        self.dz1+= dz1_res
        self.dz2+= dz2_res
        
        dWrh_res, d_hidden_res = self.deriv(self.dz2,self.Wrh, self.hidden.reshape(-1,1), '@')
        self.dWrh += dWrh_res.T
        self.dhidden +=d_hidden_res
        
        dWrx_res, dx_res = self.deriv(self.dz1, self.Wrx, self.x.reshape(-1,1), "@")
        self.dWrx += dWrx_res.T
        dx+= dx_res
        dh = self.dhidden
        
        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh

    
    def deriv(self, dz, first_term, second_term, operator):
        if operator == '+':
            deriv_first_term = dz 
            deriv_second_term = dz
        elif operator == '*':
            deriv_first_term = dz * second_term.T
            deriv_second_term = dz * first_term.T
        elif operator == '@':
            deriv_first_term = second_term @ dz
            deriv_second_term = dz @ first_term
        elif operator == '-':
            deriv_first_term = dz
            deriv_second_term = -dz
        elif operator == 'tanh':
            activation = Tanh()
            activated_value = activation(first_term)
            deriv_first_term = dz * activation.derivative().T
            deriv_second_term = None
        elif operator == 'sigmoid':
            activation = Sigmoid()
            activated_value = activation(first_term)
            deriv_first_term = dz * activation.derivative().T
            deriv_second_term =None
        else:
            raise ValueError("Did not find a operator that fit our cases")
            
        return deriv_first_term, deriv_second_term
            
