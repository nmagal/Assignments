#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:56:57 2021

@author: nicholasmagal
"""

import numpy as np
import sys
import math


class ReadInput():
    
    #Used to grab user input
    def __init__(self):
        
        self.train_path = sys.argv[1]
        self.validation_path = sys.argv[2]
        self.train_labels = sys.argv[3]
        self.validation_labels = sys.argv[4]
        self.metrics_out = sys.argv[5]
        self.num_epoch = sys.argv[6]
        self.hidden_units = sys.argv[7]
        self.weight_init = sys.argv[8]
        self.learning_rate = sys.argv[9]
    
    #Creating a dictionary to better hold user input
    def user_input_dictionary(self):
        
        user_input_dictionary = {
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "train_labels": self.train_labels,
            "validation_labels": self.validation_labels,
            "metrics_out": self.metrics_out,
            "num_epoch": self.num_epoch,
            "hidden_units": self.hidden_units,
            "weight_init": self.weight_init,
            "learning_rate": self.learning_rate
            }
        
        return(user_input_dictionary)

class DataManipulator():

    #Reads flattend picture values into a numpy array
    def import_data_to_array(self, file_to_import_from):
        
        data_set = np.genfromtxt(fname=file_to_import_from,
                                 dtype=float, delimiter=',')
        
        x_data = data_set[:, 1:]
        y_data = data_set[:,0]
        
        #Add a 1 to the start of all rows as a bias
        rows_x, cols_x = np.shape(x_data)
        bias_terms = np.ones((rows_x,1))
        x_data = np.append(bias_terms,x_data,1)
        return x_data, y_data
    
    #inits weights to either random values or all zeros (except for bias=0)
    def init_weights(self, row_dimen, col_dimen, init_mode):
        
        if init_mode == '1':
            weight_matrix = np.random.uniform(low=-.1, high=.1, 
                                              size=(int(row_dimen),int(col_dimen)-1))
            
            weight_rows, weight_col = weight_matrix.shape        
            bias_init=np.zeros((weight_rows, 1), float)
            weight_matrix = np.append(bias_init,weight_matrix, 1)
        
        if init_mode == '2':
            weight_matrix = np.zeros((int(row_dimen), int(col_dimen)))
        
        return weight_matrix

class NN():
    
    # Stochastic Gradient Descent using cross entropy as our loss function
    def sgd(self, x_training_data, alpha_weights, beta_weights, y_training_data, epochs, learning_rate, x_valid, y_valid, metrics_out_file_name):
        
        learning_rate = float(learning_rate)
        validation_average_cross_entropy = []
        training_average_cross_entropy = []
        
        for epoch_number in range(int(epochs)):
            
            for photo in range(len(x_training_data)):

                #Computing forward components of NN
                a_layer, z_layer, b_layer, soft_max_layer, y_one_hot = self.forward_propagation(x_training_data[photo],alpha_weights,beta_weights,y_training_data[photo])
                
                #Backwards Propegation 
                partial_j_wr_b, partial_j_wr_beta_weights, partial_j_wr_z, partial_j_wr_a, partial_j_wr_alpha = nn.backprop(y_one_hot, 
                                                        a_layer, 
                                                        z_layer, 
                                                        b_layer, 
                                                        soft_max_layer,
                                                        beta_weights,
                                                        x_training_data[photo],
                                                        alpha_weights)
                
                
                #Update weights 
                alpha_weights = np.subtract(alpha_weights, np.dot(learning_rate,partial_j_wr_alpha))
                beta_weights = np.subtract(beta_weights, np.dot(learning_rate, partial_j_wr_beta_weights))
            
            #Provides anyalsis on how our algorithm works
            epoch_training_cross_entropy, epoch_valid_cross_entropy = self.mean_cross_entropy(alpha_weights, 
                                                               beta_weights, 
                                                               epoch_number, 
                                                               x_training_data, 
                                                               y_training_data, 
                                                               x_valid, 
                                                               y_valid, 
                                                               metrics_out_file_name)
            
            validation_average_cross_entropy.append(epoch_valid_cross_entropy)
            training_average_cross_entropy.append(epoch_training_cross_entropy)
        
        
        #Writing out results to files
        with open(metrics_out_file_name, 'w') as f:
            for entry in range(len(validation_average_cross_entropy)):
                f.write(training_average_cross_entropy[entry])
                f.write(validation_average_cross_entropy[entry])
            

        return(alpha_weights, beta_weights)
            

    def mean_cross_entropy(self,alpha_weights, beta_weights, epoch_number, x_training_data, y_training_data, x_valid, y_valid, metrics_file_name):
        
        #Calculating Average cross entropy for training data
        total_cross_entropy_training=0.0
        
        for photo in range(len(x_training_data)):
            #Doing a forward pass so we can get our y predicted values
            a_layer, z_layer, b_layer, soft_max_layer, y_one_hot = self.forward_propagation(x_training_data[photo],alpha_weights,beta_weights,y_training_data[photo])
            photo_cross_entropy = self.cross_entropy(soft_max_layer, y_one_hot)
            total_cross_entropy_training = total_cross_entropy_training + photo_cross_entropy
        
        average_cross_entropy_training = total_cross_entropy_training/len(x_training_data)
        #print("epoch="+str(epoch_number+1)+ " crossentropy(train): "+ str(average_cross_entropy_training))
        
        total_cross_entropy_validation = 0.0
        
        for photo in range(len(x_valid)):
             a_layer, z_layer, b_layer, soft_max_layer, y_one_hot = self.forward_propagation(x_valid[photo],alpha_weights,beta_weights,y_valid[photo])
             photo_cross_entropy = self.cross_entropy(soft_max_layer, y_one_hot)
             total_cross_entropy_validation = total_cross_entropy_validation + photo_cross_entropy
        
        average_cross_entropy_validation = total_cross_entropy_validation/len(x_valid)
        #print("epoch="+str(epoch_number+1)+ " crossentropy(validation): "+ str(average_cross_entropy_validation))
 
        return("epoch="+str(epoch_number+1)+ " crossentropy(train): "+ str(average_cross_entropy_training)+"\n","epoch="+str(epoch_number+1)+ " crossentropy(validation): "+ str(average_cross_entropy_validation)+"\n")
            
    
    #Compute forward propagation. Returns computed elements of nn. Can remove cross entropy for more speed
    def forward_propagation(self, photo, alpha_weights, beta_weights, photo_class):
        
        #Our true class value in one hot encoding form
        y_one_hot = self.one_hot_encoding(photo_class)
        #a is the hidden layer before activation
        a = self.linear_forward(photo, alpha_weights)
        #z is hidden layer after activation
        z = self.sigmoid_forward(a)
        #b is the product of z and beta_weights
        b = self.linear_forward(z, beta_weights)
        softmax = self.softmax_forward(b)
        
        return(a,z,b,softmax, y_one_hot)
    
    def backprop(self, y_one_hot, a_layer, z_layer, b_layer, soft_max_layer, beta_weights, input_layer, alpha_weights):
        
        #Begin calculating Gradients
        partial_j_wr_b = self.soft_max_backward(y_one_hot, soft_max_layer)
        partial_j_wr_beta_weights, partial_j_wr_z = self.backward_linear_weights(partial_j_wr_b,
                                                                 z_layer,
                                                                 beta_weights)
        partial_j_wr_a = self.sigmoid_backwards(partial_j_wr_z,
                                                a_layer)
        
        partial_j_wr_alpha_weights = self.backward_linear_weights_final(
            partial_j_wr_a, input_layer, alpha_weights)
        
        
        return partial_j_wr_b, partial_j_wr_beta_weights,partial_j_wr_z, partial_j_wr_a, partial_j_wr_alpha_weights
        
    #Runs nuerons through fully connected weights
    def linear_forward(self, input_nuerons, weights):
        
        output_neurons = np.dot(weights,input_nuerons)
        return(output_neurons)
    
    #runs nurons through activation layer of sigmoid
    def sigmoid_forward(self,input_nuerons):
        
        activated_nuerons = np.zeros(len(input_nuerons))
        
        for nueron in range(len(input_nuerons)):
            
            activated_nueron = self.sigmoid(input_nuerons[nueron])    
            activated_nuerons[nueron] = activated_nueron
        
        #Adding a bias term with value one
        bias_term = np.ones((1,1))
        activated_nuerons_with_bias = np.append(bias_term, activated_nuerons)
            
        return(activated_nuerons_with_bias)
    
    #Calculates the probability of each possible class
    def softmax_forward(self, input_nuerons):
        
        softmax_layer = np.zeros(len(input_nuerons))
        denominator = 0
        
        for nueron in range(len(softmax_layer)):
            denominator = denominator + math.exp(input_nuerons[nueron])
        
        for nueron in range(len(softmax_layer)):
            numerator = math.exp(input_nuerons[nueron])
            softmax_layer[nueron] = numerator/denominator
        
        return softmax_layer
    
    #One hot encodes the class of a photo. 
    def one_hot_encoding(self, photo_class):
        
        one_hot_encoding = np.zeros(10)
        index_of_class=int(photo_class)
        one_hot_encoding[index_of_class]=1

        return one_hot_encoding
    
    #Calculates cross entropy of softmax output compared to one hot encoded y
    def cross_entropy(self,y_predicted, true_y_one_hot):
        
        cross_entropy = 0
        
        for nueron in range(len(y_predicted)):
            cross_entropy = cross_entropy + true_y_one_hot[nueron]*np.log(y_predicted[nueron])
        
        cross_entropy = -cross_entropy
        return cross_entropy
        
    def sigmoid(self, value):
        
        sigmoid = 1/(1+math.exp(-value))
        return sigmoid
    
    #Find the parital derivitive of J with respect of b 
    def soft_max_backward(self, true_y_one_hot, predicted_y_hat):
        
        true_y_one_hot = np.negative(true_y_one_hot)
        partial_j_respect_b = np.add(true_y_one_hot, predicted_y_hat)
        return partial_j_respect_b
    
    #Find the partial derivitive of J with respect to weights and input nueron
    def backward_linear_weights(self,partial_j_respect_b, z_layer, beta_weights):
        
        #Adding second dimension and finding partial of weights
        z_layer = np.array([z_layer])
        partial_j_respect_b = np.transpose(np.array([partial_j_respect_b]))
        partial_j_wr_beta_weights = np.dot(partial_j_respect_b,z_layer)
        
        #finding derivitive of j with respect to input nodes
        beta_weights = np.transpose(beta_weights)
        partial_of_j_wrt_z = np.dot(beta_weights,
                                         partial_j_respect_b)
        
        #Dropping the bias
        partial_of_j_wrt_z= partial_of_j_wrt_z[1:, :]
        
        
        return(partial_j_wr_beta_weights,partial_of_j_wrt_z)
    
    def backward_linear_weights_final(self, partial_j_wr_a, input_layer, alpha_weights):
        
        input_layer = np.array([input_layer])
        partial_j_wr_alpha_weights = np.dot(partial_j_wr_a, input_layer)
        
        return(partial_j_wr_alpha_weights)

    def sigmoid_backwards(self, partial_j_wr_z, a_layer):
        
        #grabbing the partial of z w.r. to a
        for nueron_number in range(len(a_layer)):
            
            nueron_value = a_layer[nueron_number]
            nueron_value_apply_derivitive = self.sigmoid(nueron_value)*((1-self.sigmoid(nueron_value)))
            
            #updating a_layer nueron 
            a_layer[nueron_number] = nueron_value_apply_derivitive
            
        #Adding second dimension to a layer
        a_layer = np.array([a_layer])
        a_layer = np.transpose(a_layer)
        
        partial_j_wr_a = a_layer * partial_j_wr_z
        
        return(partial_j_wr_a)
    
    #predicts the classes of each photo and outputs as a vector
    def predict_classes(self, x_train, alpha_weights, beta_weights, y_train, label_file_name):
        
        #Predicting classes 
        predicted_classes = []
        for photo in range(len(x_train)):
            
            photo_to_predict = x_train[photo]
            a_layer, z_layer, b_layer, softmax, y_one_hot = self.forward_propagation(photo_to_predict, alpha_weights, beta_weights, y_train[photo])
            predicted_class = np.argmax(softmax)
            predicted_classes.append(predicted_class)
        
        #Writing out predicted classes
        with open(label_file_name, 'w') as f:
            for class_value in predicted_classes:
                f.writelines(str(class_value)+ '\n')
        
        return predicted_classes
    
    #This method measures training error and writes it out to our metrics file
    def write_error_out(self, validation_predictions, training_predictions, y_train, y_validation, metrics_file_name):
        
        validation_total = len(y_validation)
        training_total = len(y_train)
        
        validation_error = 0
        
        for class_label in range(len(validation_predictions)):
            if validation_predictions[class_label] != y_validation[class_label]:
                validation_error = validation_error + 1
        
        validation_error = validation_error/validation_total
        
        training_error = 0
        
        for class_label in range(len(training_predictions)):
            if training_predictions[class_label] != y_train[class_label]:
                training_error = training_error + 1
        training_error = training_error/training_total
        
        with open(metrics_file_name, "a") as f:
            f.write("error(train): "+str(training_error) + "\n")
            f.write("error(validation): "+str(validation_error))
        


#Running code 
nn = NN()
data_manipulator = DataManipulator()
user_input = ReadInput().user_input_dictionary()

x_train, y_train = data_manipulator.import_data_to_array(
    user_input["train_path"])

x_valid, y_valid = data_manipulator.import_data_to_array(
    user_input["validation_path"])

#Alpha weights dimen will be [hidden layer x input + 1(for bias)]

x_rows, x_cols = x_train.shape 
alpha_weights = data_manipulator.init_weights(user_input["hidden_units"],
                                              x_cols, user_input["weight_init"])

#Beta weights dimen will be [softmax layer x hidden layer +1(for bias)]
beta_weights = data_manipulator.init_weights(10, 
                                             int(user_input["hidden_units"])+1,
                                             user_input["weight_init"])

#Performing Stochastic Gradient Descent 
updated_alpha_weights, updated_beta_weights = nn.sgd(x_train, alpha_weights, beta_weights, y_train, user_input["num_epoch"], user_input["learning_rate"], x_valid, y_valid,user_input["metrics_out"])

#Predicting and writing out training predictions
predicted_training = nn.predict_classes(x_train, updated_alpha_weights,updated_beta_weights,y_train,user_input["train_labels"])

#Predicting and writing out validation predictions
predicted_validation = nn.predict_classes(x_valid, updated_alpha_weights,updated_beta_weights,y_valid,user_input["validation_labels"])

#Calculating and writing train and validation error
nn.write_error_out(predicted_validation, predicted_training, y_train, y_valid, user_input["metrics_out"])



                                             
