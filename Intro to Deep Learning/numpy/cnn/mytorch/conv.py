# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        
        self.db = np.zeros(self.b.shape)
        self.x = np.zeros(None)
        self.convolved_output = np.zeros(None)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        
        self.x = x
        input_batch_dimen, input_in_channel_dimen, input_width = x.shape
        
        #Calculating ouput dimensions
        output_width = ((input_width - self.kernel_size)//self.stride) + 1
        convolved_output = np.zeros((input_batch_dimen,self.out_channel, output_width))
        
        
        for starting_sliding_window_index in range(output_width):
            input_window = x[:,:, starting_sliding_window_index*self.stride:starting_sliding_window_index*self.stride+self.kernel_size]
            
            #Convolving with tensordot
            convolved_output[:,:, starting_sliding_window_index] = np.tensordot(input_window,self.W, axes=((1,2),(1,2))) 
        
        #Adding Bias
        for output_channel_index in range(self.out_channel):
            convolved_output[:, output_channel_index,:] = convolved_output[:, output_channel_index,:] + self.b[output_channel_index]
    
        self.y = convolved_output
        return(convolved_output)
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        dx = np.zeros(self.x.shape)
        orginal_delta = delta
        
        #Finding the derivitive wr W
        if self.stride > 1:
            delta = self.oneD_dialation(delta,self.stride)
            
        batch_delta_shape, channel_delta_shape, output_delta_length = delta.shape
        input_batch_dimen, input_in_channel_dimen, input_width = self.x.shape        
        dw_length = ((input_width - output_delta_length)) + 1
        
        for starting_sliding_window_index in range(self.kernel_size):
            input_window = self.x[:, :, starting_sliding_window_index:starting_sliding_window_index+output_delta_length]
            #self.dW should be out channel x in_channel x dw_length (we modify dw with dialation if stride, this ends up being filter length)
            self.dW[:, :,starting_sliding_window_index]  = np.tensordot(delta,input_window, axes=((0,2),(0,2)))           

        
        #Finding the derivitive wr b
        self.db = np.sum(delta, axis=(0,2))
        
        #Finding the derivative wr X
        flipped_kernal = self.W[:,:,::-1]
        x,y,flipped_kernal_width = flipped_kernal.shape
        
        amount_to_pad = flipped_kernal_width - 1 
        padded_delta = orginal_delta
        
        #If stride is greater then 1 we must modify delta
        if self.stride > 1:
            padded_delta=self.oneD_dialation(padded_delta,self.stride)
        padded_delta = self.padder(padded_delta,amount_to_pad)
        
        batch_padded_delta_dimen, output_padded_delta_dimen, width_padded_delta_dimen = padded_delta.shape  
        dx_length = (width_padded_delta_dimen-flipped_kernal_width)+1
               
        for starting_sliding_window_index in range(dx_length):
            input_window = padded_delta[:, :, starting_sliding_window_index:starting_sliding_window_index+flipped_kernal_width]
            
            #convolving flipped kernal with input, this gives us our dx 
            dx[:,:,starting_sliding_window_index] = np.tensordot(input_window,flipped_kernal, axes=((1,2),(0,2)))
        
        return(dx)
    
    #If we have a stride greater then 1, for backprop we must dialate 
    def oneD_dialation(self,matrix,stride):
        batch_size, channel_size, length_size = matrix.shape
        columns_to_append = np.zeros(((length_size-1)*(stride-1)))
        columns_to_append = columns_to_append.reshape(-1,(stride-1))
        
        matrix_index = 1
    
        for dilator_group in columns_to_append:
            dialtor_index = 0
            for dilator_instance in dilator_group:
                matrix = np.insert(matrix, matrix_index+dialtor_index, dilator_instance, axis=2)
                dialtor_index = dialtor_index + 1
                
            matrix_index = matrix_index + stride
        
        return(matrix)
    
    #If we are calcualting d wr x, we must pad the filter to account for some x not effecting multiple values
    def padder(self, matrix_to_pad, pad_amount):
        
        pad_values = np.zeros((pad_amount))

        for padder in pad_values:        
            matrix_to_pad = np.insert(matrix_to_pad, 0, padder, axis=2)
        
        x,y,z = matrix_to_pad.shape
        pad_values = np.zeros((x,y,pad_amount))
        matrix_to_pad = np.append(matrix_to_pad, pad_values, axis=2)
        
        return(matrix_to_pad)


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.x = np.zeros(None)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        input_batch_size, input_channel_size, input_width_size, input_height_size = x.shape
        
        output_width = ((input_width_size-self.kernel_size)//self.stride) + 1
        output_height = ((input_height_size-self.kernel_size)//self.stride) + 1
        output = np.zeros((input_batch_size,self.out_channel,output_width,output_height))
        
        for sliding_window_height in range(output_height):
            for sliding_window_width in range(output_width):
                input_window = x[:,:, sliding_window_width*self.stride:sliding_window_width*self.stride+self.kernel_size, sliding_window_height*self.stride:sliding_window_height*self.stride+self.kernel_size]
                #Convolving with tensordot
                output[:,:,sliding_window_width,sliding_window_height] = np.tensordot(input_window,self.W, axes=((1,2,3),(1,2,3))) + self.b
        
        
        return(output)
    
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx = np.zeros(self.x.shape)
        orginal_delta = delta
        
        #Finding dr wr. the filter
        if self.stride > 1:
            delta = self.twoD_dilation(delta,self.stride,2)
            delta = self.twoD_dilation(delta, self.stride,3)
        
        batch_delta_shape, channel_delta_shape, width_delta, height_delta = delta.shape
        input_batch_dimen, input_in_channel_dimen, input_width, height_input = self.x.shape  
        
        dw_length = ((input_width - width_delta)) + 1
        dw_height = ((height_input - height_delta)) + 1
        
        for sliding_window_height in range (self.kernel_size):
            for sliding_window_width in range (self.kernel_size):
                input_window = self.x[:,:, sliding_window_width:sliding_window_width+width_delta, sliding_window_height:sliding_window_height+height_delta]
                self.dW[:,:, sliding_window_width, sliding_window_height] = np.tensordot(delta,input_window, axes=((0,2,3),(0,2,3)))
        
        #Finding the derivative wr. b
        self.db = np.sum(delta, axis=(0,2,3))
        
        #Finding the derivative w.r to x
        flipped_kernal = np.rot90(self.W,2,axes=(2,3))
        x,y,flipped_kernal_width, flipped_kernal_height = flipped_kernal.shape
        
        amount_to_pad_width = flipped_kernal_width-1
        amount_to_pad_height = flipped_kernal_height-1
        
        #setting this so we modify the orginal delta
        padded_delta = orginal_delta
        
        #Dialating delta to account for stride
        if self.stride > 1:
            padded_delta = self.twoD_dilation(padded_delta,self.stride, 2)
            padded_delta = self.twoD_dilation(padded_delta,self.stride, 3)
        
        padded_delta = self.twoD_padder(padded_delta,amount_to_pad_width,3)
        padded_delta = self.twoD_padder(padded_delta,amount_to_pad_height,2)
        
        batch_padded_delta_dimen, output_padded_delta_dimen, width_padded_delta_dimen, height_padded_delta_dimen = padded_delta.shape 
        dx_length = (width_padded_delta_dimen-flipped_kernal_width)+1
        dx_height = (height_padded_delta_dimen - flipped_kernal_height)+1
        
        for sliding_window_height in range (dx_height):
            for sliding_window_width in range (dx_length):
                input_window = padded_delta[:, :, sliding_window_width:sliding_window_width+flipped_kernal_width, sliding_window_height:sliding_window_height+flipped_kernal_height]
                dx[:,:,sliding_window_width,sliding_window_height] = np.tensordot(input_window, flipped_kernal, axes=((1,2,3),(0,2,3)))
        
        return(dx)
    
    #This will dialte a matrix depending on which axis is given
    def twoD_dilation(self, matrix_to_dilate, stride, axis):
        x,y,length,height = matrix_to_dilate.shape
        
        if axis==2:
            columns_to_append = np.zeros(((length - 1) * (stride-1)))
            columns_to_append = columns_to_append.reshape(-1,(stride-1))
            
        if axis==3:
            columns_to_append = np.zeros(((height- 1) * (stride-1)))
            columns_to_append = columns_to_append.reshape(-1,(stride-1))
        
        matrix_index = 1
    
        for dilator_group in columns_to_append:
            dialtor_index = 0
            for dilator_instance in dilator_group:
                matrix_to_dilate = np.insert(matrix_to_dilate, matrix_index+dialtor_index, dilator_instance, axis=axis)
                
                dialtor_index = dialtor_index + 1
                
            matrix_index = matrix_index + stride
        
        return(matrix_to_dilate) 
    
    #If we are calcualting d wr x, we must pad the filter to account for some x not effecting multiple values
    def twoD_padder(self, matrix_to_pad, pad_amount,axis):
        
        pad_values = np.zeros((pad_amount))
    
        for padder in pad_values:        
            matrix_to_pad = np.insert(matrix_to_pad, 0, padder, axis=axis)
        
        x,y,width,height = matrix_to_pad.shape
        if axis==2:
            pad_values = np.zeros((x,y,pad_amount,height))
        if axis==3:
            pad_values = np.zeros((x,y,width,pad_amount))
        matrix_to_pad = np.append(matrix_to_pad, pad_values, axis=axis) 
        return(matrix_to_pad)


class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.kernel_dilated = (kernel_size - 1)*(dilation-1) + kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.dW_dilated = np.zeros(self.W_dilated.shape)
        self.db = np.zeros(self.b.shape)
        self.x = np.zeros(None)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        
        x = self.twoD_padder(x, self.padding, axis =2)
        x = self.twoD_padder(x, self.padding, axis =3)
        self.x = x

        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        self.W_dilated = self.twoD_dilation(self.W,self.dilation,2)
        self.W_dilated = self.twoD_dilation(self.W_dilated,self.dilation,3)
        
        # TODO: regular forward, just like Conv2d().forward()
        input_batch_size, input_channel_size, input_width_size, input_height_size = x.shape
        
        output_width = ((input_width_size-self.kernel_dilated)//self.stride) + 1
        output_height = ((input_height_size-self.kernel_dilated)//self.stride) + 1
        output = np.zeros((input_batch_size,self.out_channel,output_width,output_height))
        
        for sliding_window_height in range(output_height):
            for sliding_window_width in range(output_width):
                input_window = x[:,:, sliding_window_width*self.stride:sliding_window_width*self.stride+self.kernel_dilated, sliding_window_height*self.stride:sliding_window_height*self.stride+self.kernel_dilated]
                output[:,:,sliding_window_width,sliding_window_height] = np.tensordot(input_window,self.W_dilated, axes=((1,2,3),(1,2,3))) + self.b
        
        return(output)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # TODO: main part is like Conv2d().backward(). The only difference are: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.
        
        
        dx = np.zeros(self.x.shape)
        orginal_delta = delta
        
        #Finding dr wr. the filter
        if self.stride > 1:
            delta = self.twoD_dilation(delta,self.stride,2)
            delta = self.twoD_dilation(delta, self.stride,3)
        
        batch_delta_shape, channel_delta_shape, width_delta, height_delta = delta.shape
        input_batch_dimen, input_in_channel_dimen, input_width, height_input = self.x.shape  
        
        dw_length = ((input_width - width_delta)) + 1
        dw_height = ((height_input - height_delta)) + 1
        
        for sliding_window_height in range (self.kernel_dilated):
            for sliding_window_width in range (self.kernel_dilated):
                input_window = self.x[:,:, sliding_window_width:sliding_window_width+width_delta, sliding_window_height:sliding_window_height+height_delta]
                self.dW_dilated[:,:, sliding_window_width, sliding_window_height] = np.tensordot(delta,input_window, axes=((0,2,3),(0,2,3)))
        
        self.dW = self.undialate(self.dW_dilated, self.dilation)
        
        #Finding the derivative wr. b
        self.db = np.sum(delta, axis=(0,2,3))
        
        #Finding the derivative w.r to x  (if have bugs check this is the right way to go)
        flipped_kernal = np.rot90(self.W_dilated,2,axes=(2,3))
        x,y,flipped_kernal_width, flipped_kernal_height = flipped_kernal.shape
        
        amount_to_pad_width = flipped_kernal_width-1
        amount_to_pad_height = flipped_kernal_height-1
        
        #setting this so we modify the orginal delta
        padded_delta = orginal_delta
        
        #Dialating delta to account for stride
        if self.stride > 1:
            padded_delta = self.twoD_dilation(padded_delta,self.stride, 2)
            padded_delta = self.twoD_dilation(padded_delta,self.stride, 3)
        
        padded_delta = self.twoD_padder(padded_delta,amount_to_pad_width,3)
        padded_delta = self.twoD_padder(padded_delta,amount_to_pad_height,2)
        
        batch_padded_delta_dimen, output_padded_delta_dimen, width_padded_delta_dimen, height_padded_delta_dimen = padded_delta.shape 
        dx_length = (width_padded_delta_dimen-flipped_kernal_width)+1
        dx_height = (height_padded_delta_dimen - flipped_kernal_height)+1
        
        for sliding_window_height in range (dx_height):
            for sliding_window_width in range (dx_length):
                input_window = padded_delta[:, :, sliding_window_width:sliding_window_width+flipped_kernal_width, sliding_window_height:sliding_window_height+flipped_kernal_height]
                dx[:,:,sliding_window_width,sliding_window_height] = np.tensordot(input_window, flipped_kernal, axes=((1,2,3),(0,2,3)))
        
        #Have to unpad to account for dialation 
        dx = self.unpadder(dx, self.padding)
        return(dx)
    

    def twoD_padder(self, matrix_to_pad, pad_amount,axis):
        
        pad_values = np.zeros((pad_amount))
    
        for padder in pad_values:        
            matrix_to_pad = np.insert(matrix_to_pad, 0, padder, axis=axis)
        
        x,y,width,height = matrix_to_pad.shape
        if axis==2:
            pad_values = np.zeros((x,y,pad_amount,height))
        if axis==3:
            pad_values = np.zeros((x,y,width,pad_amount))
        matrix_to_pad = np.append(matrix_to_pad, pad_values, axis=axis) 
        return(matrix_to_pad)
    
    #This will dialte a matrix depending on which axis is given
    def twoD_dilation(self, matrix_to_dilate, stride, axis):
        x,y,length,height = matrix_to_dilate.shape
        
        if axis==2:
            columns_to_append = np.zeros(((length - 1) * (stride-1)))
            columns_to_append = columns_to_append.reshape(-1,(stride-1))
            
        if axis==3:
            columns_to_append = np.zeros(((height- 1) * (stride-1)))
            columns_to_append = columns_to_append.reshape(-1,(stride-1))
        
        matrix_index = 1
    
        for dilator_group in columns_to_append:
            dialtor_index = 0
            for dilator_instance in dilator_group:
                matrix_to_dilate = np.insert(matrix_to_dilate, matrix_index+dialtor_index, dilator_instance, axis=axis)
                
                dialtor_index = dialtor_index + 1
                
            matrix_index = matrix_index + stride
        
        return(matrix_to_dilate) 
    
    def undialate(self, matrix_to_undilate, dilation_amount):
    
        x,y,height, width = matrix_to_undilate.shape
        groups_to_delete_width = width//dilation_amount
        groups_to_delete_height = height//dilation_amount
        
        
        #We create two for loops here. One for getting slices to delete, and one for getting index of slices.
        indexes_to_delete_width=[]
        moving_index = 1
        for group_to_delete in range(groups_to_delete_width):
            indexes_to_slice = np.s_[moving_index:moving_index+dilation_amount-1]
            
            moving_index_2=0
            for index in range(indexes_to_slice.stop-indexes_to_slice.start):
                    indexes_to_delete_width.append(indexes_to_slice.start+moving_index_2)
                    moving_index_2 = moving_index_2 +1
    
            moving_index = moving_index+dilation_amount
        
        #Now must do the same for the height 
        indexes_to_delete_height=[]
        moving_index = 1
        for group_to_delete in range(groups_to_delete_height):
            indexes_to_slice = np.s_[moving_index:moving_index+dilation_amount-1]
            
            moving_index_2=0
            for index in range(indexes_to_slice.stop-indexes_to_slice.start):
                    indexes_to_delete_height.append(indexes_to_slice.start+moving_index_2)
                    moving_index_2 = moving_index_2 +1
    
            moving_index = moving_index+dilation_amount    
            
            
        matrix_to_undilate = np.delete(matrix_to_undilate,indexes_to_delete_width, axis=3)
        matrix_undilated = np.delete(matrix_to_undilate,indexes_to_delete_height, axis=2)

            
        return(matrix_undilated)
    
    def unpadder(self, matrix_to_unpad, unpad_amount):
        x,y,height,width = matrix_to_unpad.shape
        top_index_unpad = np.arange(unpad_amount)
        bottem_index_unpad = (top_index_unpad +1) *-1
        #Autograder runs on old np that does not delete negative indexes, must convert these to positive
        bottem_index_unpad = width + bottem_index_unpad
        full_indexes_unpad = np.concatenate((top_index_unpad, bottem_index_unpad))
        #Delete front rows and columns
        matrix_to_unpad = np.delete(matrix_to_unpad, full_indexes_unpad, axis=2)
        matrix_to_unpad = np.delete(matrix_to_unpad, full_indexes_unpad, axis=3)
        return(matrix_to_unpad)
    

            

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        flattened_layer = x.reshape(self.b,self.c * self.w)
        return(flattened_layer)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        
        flattened_layer_dx_reshaped = np.reshape(delta, (self.b, self.c, self.w))
        return(flattened_layer_dx_reshaped)


