#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:42:21 2021

@author: nicholasmagal
"""
import numpy as np
import sys
import math

class NaiveBayes():

    def import_data_to_array_split(self, file_to_import_from):
        
        data_set = np.genfromtxt(fname=file_to_import_from,
                                 dtype= str, delimiter=',')
        
        x_train = data_set[1:, :-1]
        y_train = data_set[1:,-1]
        
        return(x_train, y_train)
        

    #Reads flattend picture values into a numpy array
    def import_data_to_array(self, file_to_import_from):
        
        data_set = np.genfromtxt(fname=file_to_import_from,
                                 dtype= str, delimiter=',')
        return data_set
    
    #Creates mean and variance tables. Can be viewed as 'training' 
    def create_mean_and_variance_tables(self, data_set,features_to_use):
        
        tool_data = data_set[np.where(data_set[:,-1] == 'tool')]
        building_data = data_set[np.where(data_set[:,-1] == 'building')]
        
        #Dropping label
        tool_data = tool_data[: , :-1]
        building_data = building_data[ : ,:-1]
        
        building_probability = building_data.shape[0]/data_set.shape[0]
        tool_probabilty = tool_data.shape[0]/data_set.shape[0]
        
        tool_data = tool_data.astype(float)
        building_data = building_data.astype(float)
        
        tool_data_mean = np.mean(tool_data, axis=0)
        tool_data_mean= tool_data_mean.reshape(1,len(tool_data_mean))
        
        tool_data_variance = np.var(tool_data, axis = 0)
        tool_data_variance = tool_data_variance.reshape(1,len(tool_data_variance))
        
        building_data_mean = np.mean(building_data, axis=0)
        building_data_mean= building_data_mean.reshape(1,len(building_data_mean))
        
        building_data_variance = np.var(building_data, axis = 0)
        building_data_variance = building_data_variance.reshape(1,len(building_data_variance))
        
        #In order to optimize model, we will drop features that have similar means
        tools_minus_buildings_mean = np.absolute(tool_data_mean - building_data_mean)
        indexes_to_drop = tools_minus_buildings_mean[0].argsort()[:len(tools_minus_buildings_mean[0])-int(features_to_use)]
        
        #Dropping those indexes
        tool_data_mean = np.delete(tool_data_mean,indexes_to_drop, axis = 1 )
        tool_data_variance = np.delete(tool_data_variance,indexes_to_drop, axis = 1 )
        building_data_mean = np.delete(building_data_mean,indexes_to_drop, axis = 1 )
        building_data_variance = np.delete(building_data_variance,indexes_to_drop, axis = 1 )
        
        
        #Creating model
        naive_bayes_model = {"tool_mean": tool_data_mean,
                                "tool_variance": tool_data_variance,
                                "building_data_mean": building_data_mean,
                                "building_data_variance":building_data_variance,
                                "building_prob" : building_probability,
                                "tool_prob": tool_probabilty}
        
        return (naive_bayes_model, indexes_to_drop)
    
    #Given mean, variance, and a data point calculate the probability using the normal distribution 
    def normal_distribution_pdf(self,mean, variance, point):
        point = float(point)
        left_hand_equation = 1/(math.sqrt(2 * math.pi *variance))
        right_hand_equation = math.exp(-(point - mean)**2/(2*variance))
        probability = left_hand_equation * right_hand_equation
        return(probability)
    
    def classify(self,niave_bayes_model,data):
        
        classification_labels = []
        
        for row in data:
            
            #used to hold probabilities that we will then sum
            tool_log_probabilities = []
            building_log_probabilities = []
            
            for pixel in range(len(row)):
                
                #Calculating probability for tools
                mean_tool = niave_bayes_model["tool_mean"][0][pixel]
                variance_tool = naive_bayes_model["tool_variance"][0][pixel]
                pixel_tool_probability = np.log(self.normal_distribution_pdf(mean_tool,variance_tool,
                                                       row[pixel]))
                tool_log_probabilities.append(pixel_tool_probability)
                
                #Calculating probability for building
                mean_building = niave_bayes_model["building_data_mean"][0][pixel]
                variance_building = naive_bayes_model["building_data_variance"][0][pixel]
                pixel_building_probability = np.log(self.normal_distribution_pdf(mean_building,variance_building,
                                                       row[pixel]))
                building_log_probabilities.append(pixel_building_probability)
            
            picture_tool_probability = np.log(naive_bayes_model["tool_prob"]) +np.sum(tool_log_probabilities)
            picture_building_probability = np.log(naive_bayes_model["building_prob"]) +np.sum(building_log_probabilities)
            
            
            #Classification
            if picture_tool_probability > picture_building_probability:
                classification_labels.append("tool")
            else:
                classification_labels.append("building")
            
        return(classification_labels)
    
    #Used for writing out inference 
    def write_labels(self, training_classifications, testing_classifications, training_label_name, testing_label_name):
        
        with open(training_label_name, "w") as f: 
            for class_value in training_classifications:
                f.writelines(class_value+"\n")
        with open(testing_label_name, "w") as f: 
            for class_value in testing_classifications:
                f.writelines(class_value+"\n")
    
    #Used for writing out inference 
    def write_metrics(self, training_classifications, testing_classifications, y_train, y_test, metrics_filename):
        
        training_error = 0
        testing_error = 0
        
        for predicted_class in range(len(training_classifications)):
            
            if training_classifications[predicted_class] != y_train[predicted_class]:
                training_error = training_error +1 
        
        training_error = training_error/len(training_classifications)
        
        for predicted_class in range(len(testing_classifications)):
            
            if testing_classifications[predicted_class] != y_test[predicted_class]:
                testing_error = testing_error + 1
        
        testing_error = testing_error/len(testing_classifications)
        
        with open(metrics_filename, 'w') as f:
            f.writelines("error(train): "+str(training_error)+ "\n")
            f.writelines(("error(test: "+str(testing_error)+ "\n"))
            
        
        

            
if __name__ == '__main__':
            
        
    #User input order is as follows: training data, testing data, train labels, test labels, metrics, number of voxels
    user_input = sys.argv
    
    naive_bayes = NaiveBayes()
    
    x_train, y_train = naive_bayes.import_data_to_array_split(user_input[1])
    x_test, y_test = naive_bayes.import_data_to_array_split(user_input[2])
    

    data_set = naive_bayes.import_data_to_array(user_input[1])
    naive_bayes_model, indexes_to_drop = naive_bayes.create_mean_and_variance_tables(data_set,user_input[6])
    
    #Dropping the indexes from our data that we are not using to optimize model
    x_train = np.delete(x_train, indexes_to_drop, axis = 1)
    x_test = np.delete(x_test, indexes_to_drop, axis = 1 )
    
    #Classifying based off of our model
    train_classification_labels = naive_bayes.classify(naive_bayes_model, x_train)
    test_classification_labels = naive_bayes.classify(naive_bayes_model, x_test)
    
    #Writing out results 
    naive_bayes.write_labels(train_classification_labels, test_classification_labels
                             ,user_input[3], user_input[4])
    naive_bayes.write_metrics(train_classification_labels,
                              test_classification_labels,
                              y_train, y_test,
                              user_input[5])