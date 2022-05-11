#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 20:24:38 2021

@author: nicholasmagal

#Binary decision tree
"""

import sys
import numpy as np

#Node class that contains one node of the decision tree 
class Node():
    
    def __init__(self, data):
                
        #All the samples in the node
        self.data = data
        
        #The attribute value of the node
        self.attribute = None
        
        #Setting the left branch that correlates to No
        self.left= None
        
        #Setting the right branch that correlates to Yes
        self.right = None 
        
        self.leaf_value = None
        
        self.depth = None

class classifierOutputs():
    
    #Populates the label file with classification 
    def create_label_file_content(self, node, row, attribute_value_one, 
                          attribute_value_two, file_to_write_to):
               
        #Continue going down the tree until we reach a leaf
        if node.leaf_value == None:
            
            #Finding the attribute of the node 
            attribute_to_check = node.attribute
            
            #if the rows data is attribute_one, recurse down the left branch
            if row[attribute_to_check] == attribute_value_one:
               self.create_label_file_content(node.left, row, attribute_value_one,
                                      attribute_value_two, file_to_write_to)
               
            #otherwise go down the other branch
            else:
                self.create_label_file_content(node.right, row, attribute_value_one,
                                       attribute_value_two, file_to_write_to)
        #Once we hit this, this means we hit our classification, so write this out
        else:
             predicted_class = node.leaf_value
             
             with open(file_to_write_to, 'a') as f:
                 f.writelines(predicted_class + "\n")
    
    def create_blank_initial_file(self, file_name):
        
        with open(file_name, 'w') as f:
            pass
    
    #Method for figuring out how accuracte the classifiers were 
    def obtain_classification_error(self, data, predicted_classification_file):
        
        #Create a list of all of the classification values based off of data
        class_values = data[:,-1]
        
        #Create a list of all of the predicted classification values off of data
        predicted_class_values = np.genfromtxt(fname=predicted_classification_file, dtype=str, delimiter='\n')
        
        #Total to be used to calculate error of the classifier
        total_number_of_data = len(predicted_class_values)
        
        #Create a loop to go through class labels and see if they match up    
        index = 0
        total_error = 0
        
        for classification_label in predicted_class_values:
            if classification_label != class_values[index]:
                total_error = total_error + 1
            index = index + 1
        
        error = total_error/total_number_of_data
        
        return(error)  

    def print_metric_file(self, training_data_error,testing_data_error,
                          metric_file_name):
        #Creating a fresh file that overwrites any old previous file
        with open(metric_file_name, 'w') as f:
            f.writelines("error(train): " +str(training_data_error) + '\n')
            f.writelines("error(test): " + str(testing_data_error))          

#Class used to grow our descision tree 
class growTree():
    
    def __init__(self):
        pass
    
    def majority_vote(self, data, node,class_label_one,class_label_two,
                      attribute_index):
        data_manipulator = dataManipulator()
        #Returns the correct leaf output based off of majority of cases
        class_one, class_two = data_manipulator.value_distribution(data, 
                                                                   class_label_one, 
                                                                   class_label_two, 
                                                                   attribute_index)
        
        if class_one > class_two:
            node.leaf_value = class_label_one
            
        else:
            node.leaf_value = class_label_two
            
        return(node)
        
    
    def leaf_labeler(self,training_data, class_label_one, class_label_two):
        
        #Size of numpy array used for figuring out where to evaluate for the label
        num_rows, num_cols = training_data.shape
        #Create two seperate ares with different classifier values
        class_one_data_set = training_data[np.where(training_data[:,num_cols-1] == class_label_one )]
        class_two_data_set = training_data[np.where(training_data[:,num_cols-1] == class_label_two )]
        
        #finding which one is greater
        class_one_rows, class_one_columns = class_one_data_set.shape
        class_two_rows, class_two_columns = class_two_data_set.shape
        
        if class_one_rows > class_two_rows:
            return(class_label_one)
        elif class_one_rows < class_two_rows:
            return(class_label_two)
        else:
            #Ties should return label that comes later
            first_letter_of_class_one_label = class_label_one[0]
            first_letter_of_class_two_label = class_label_two[0]
            
            if first_letter_of_class_one_label < first_letter_of_class_two_label:
               
                return class_label_two
            
            else:
                
                return class_label_one
    
    #Function used to recursivily grow a tree
    def grow(self, node, max_depth, class_label_one, class_label_two, 
             attribute_total, attribute_values_strings, attribute_value_one,
             attribute_value_two):
        
        #Create tools
        data_api = dataManipulator()
        information_tools = informationGain()
        
        #Get a list of all attributes
        attribute_list = data_api.attribute_list(node.data)
                
        #From this list, calculate which attribute we should split on
        attribute_to_split_on = information_tools.highest_information_gain_seeker(attribute_list,
                                                                                  class_label_one,
                                                                                  class_label_two)
               
        #Base cases for when to stop building tree, may need to add one more for when its unambigious
        if node.depth < max_depth and attribute_to_split_on[0,1]!= 0 and node.depth < attribute_total:
              
            #For printing the tree
            index_of_attribute = attribute_to_split_on[0,0]
            index_of_attribute = int(index_of_attribute)
            
            #setting this node's attribute that is being used to split data
            node.attribute = index_of_attribute
            
            #Creating the new data to be pased along to the new nodes
            value_one_data_frame, value_two_data_frame = data_api.divide_data_based_off_attribute_value(node.data, 
                                                                                                        attribute_value_one, 
                                                                                                        attribute_value_two, 
                                                                                                        index_of_attribute)
            
            #Grabbing the total number of data, based off the splits for each split
            dataframe1_v1_class, dataframe1_v2_class = data_api.value_distribution(value_one_data_frame,
                                                                                   class_label_one,
                                                                                   class_label_two,
                                                                                   -1)
            
            #Grabbing the total number of data, based off the splits for each split
            dataframe2_v1_class, dataframe2_class = data_api.value_distribution(value_two_data_frame,
                                                                                   class_label_one,
                                                                                   class_label_two,
                                                                                   -1)
            
            print("|" * (node.depth +1) + " " + attribute_values_strings[index_of_attribute]
                  +" = " + attribute_value_one + ": ["+str(dataframe1_v1_class)+" "+class_label_one+"/"
                  +str(dataframe1_v2_class)+" "+ class_label_two+"]")
            
            
            #Recursivly call the function to keep growing on the left of the tree
            node.left = Node(value_one_data_frame)
            node.left.depth = node.depth + 1
            self.grow(node.left, max_depth, class_label_one, class_label_two, attribute_total,
                      attribute_values_strings,attribute_value_one,attribute_value_two)
            
            print("|" * (node.depth +1) + " " + attribute_values_strings[index_of_attribute]
                  +" = " + attribute_value_two + ": ["+str(dataframe2_v1_class)+" "+class_label_one+"/"
                  +str(dataframe2_class)+" "+ class_label_two+"]")
        
            #Recurse call function to grow on the right of the tree
            node.right = Node(value_two_data_frame)
            node.right.depth = node.depth + 1
            self.grow(node.right, max_depth, class_label_one, class_label_two, attribute_total,
                      attribute_values_strings,attribute_value_one,attribute_value_two)      
            
            return(node)
        
        else:
            
            node.leaf_value = self.leaf_labeler(node.data, class_label_one, class_label_two)

class readInput():
    
    #used to grab all userinput initially 
    def __init__(self):
        
        self.training_data = sys.argv[1]
        self.testing_data = sys.argv[2]
        self.max_tree_depth = sys.argv[3]
        self.train_labels_file_name = sys.argv[4]
        self.test_labels_file_name = sys.argv[5]
        self.metrics_file_name = sys.argv[6]
    
    #Creating a dictionary to better hold user input
    def user_input_dictionary(self):
        
        user_input_dictionary = {
            "training_data": self.training_data,
            "testing_data": self.testing_data,
            "max_tree_depth": self.max_tree_depth,
            "train_labels_file_name": self.train_labels_file_name,
            "test_labels_file_name": self.test_labels_file_name,
            "metrics_file_name": self.metrics_file_name
            }
        
        return(user_input_dictionary)

class dataManipulator():
    
    def __init__(self):
        pass
    
    #helper funciton that returns attributes in strings
    def attributes_in_strings_list(self, data):
        attribute_names = data[0]
        return attribute_names 
    
    #imports data into nummpy array
    def import_data_to_array(self, file_name):
        
        #Data set containing rows and labels of data based off of binary attributes 
        data_set = np.genfromtxt(fname=file_name, dtype=str, delimiter='\t')
        raw_data_set = np.genfromtxt(fname=file_name, dtype=str, delimiter='\t')
        #Trim off top row, unneeded labels
        data_set = np.delete(data_set,(0), axis=0)
        
        return data_set, raw_data_set
    
    #This method returns how much of the data belongs to value_one and value_two
    def value_distribution(self,data,feature_value_one,feature_value_two,attribute_index):
        
        dataset_value_one, dataset_value_two = self.divide_data_based_off_attribute_value(data, 
                                                                       feature_value_one, 
                                                                       feature_value_two, 
                                                                       attribute_index)
        
        #figure out how many of these data sets have class_one values and class_two_values
        rows_data_one, col_data_one = dataset_value_one.shape
        rows_data_two, col_data_two = dataset_value_two.shape
        
        return rows_data_one, rows_data_two,
    
    #splits all attributes into a list of attribute_nodes 
    def attribute_list(self, data):
        
        #Grabbing shape of data to be able to iterate over data
        rows, columns = data.shape
        
        #Creating a list to hold attribute_nodes
        attribute_nodes = []
        
        for x in range(columns):
            
            #We do not want to make a attribute_node for the class, so skip the last column corresponding to class  
            if x == (columns - 1):
                break
            
            #Create a new attribute node
            new_attribute_data = data[:, [x, -1]]
            new_attribute_node = attribute_node(new_attribute_data)
            attribute_nodes.append(new_attribute_node)
    
        return(attribute_nodes)
    
    #Returns two sets of data, one that has the value of the attribute and the other that does not 
    def divide_data_based_off_attribute_value(self, training_data, feature_value_one, 
                                              feature_value_two, attribute_index):
                
        data_with_feature_one = training_data[np.where(training_data[:,attribute_index] == feature_value_one )]
        
        data_with_feature_two = training_data[np.where(training_data[:,attribute_index] == feature_value_two )]
        
        return (data_with_feature_one, data_with_feature_two)
    
    #get unique attribute values
    def attribute_values(self, data, attribute_index):
        
        #Iterate through rows looking at class labels until two unique values are found 
        value_one = data[0][attribute_index]
        
        for row in data:
            
            value_two = row[attribute_index]

            #If they don't equal the same, you have found your two binary classes
            if value_two != value_one:
                
                break
            
        return(value_one, value_two)
    
    def class_distribution_printer(self,data, class_one_label, class_two_label):
        
        #Calculate the total of each class
        class_one_total = 0
        class_two_total = 0
        
        for row in data:
            if row[-1] == class_one_label:
                class_one_total = class_one_total +1
            else:
                class_two_total = class_two_total +1
        
        print("[" + str(class_one_total) +" "+
              class_one_label+" / "+str(class_two_total) +" "+class_two_label + "]")
    
#Class used to hold attribute data points corresponding to class labels, and information gain   
class attribute_node:
    
    def __init__(self, data):
        
        #Init each IG as none
        self.imformation_gain = None 
        
        #data should be a 2 x n matrix, one row for attribute values, the other for class values
        self.data = data
     
            
#Class used to calculate Information Gain. Information Gain is used to choose which attribute to split on. 
class informationGain():
    
    def __init__(self):
        pass
    
    #Method used to calculate entropy of a dataset
    def calculate_entropy(self, data, label_one, label_two):
        
        #Seperate the arrays based off of class value
        data_api=dataManipulator()
        #Creating data frames with seperate class values
        class_one_value, class_two_value = data_api.divide_data_based_off_attribute_value(
            data, label_one, label_two, -1)
        
        #Count the total amount of data from each class
        class_one_rows, class_one_columns = class_one_value.shape
        class_two_rows, class_two_columns = class_two_value.shape
        
        #Calculating variable values for use on entropy formula
        
        total = class_one_rows + class_two_rows
        
        percent_class_one = class_one_rows/total
        
        percent_class_two = class_two_rows/total
        
        #If entropy is 0, return now to avoid dividing by zero
        if (percent_class_one == 0 or percent_class_two == 0 ):
            return 0
        
        #Calculate entropy 
        entropy = -( percent_class_one * np.log2(percent_class_one) + percent_class_two * np.log2(percent_class_two))
        
        return entropy
    
    #Function that returns the attribute with the highest info_gain and info gain value
    def highest_information_gain_seeker(self, attribute_nodes, 
                                        label_one_value, label_two_value):
        
        #First value coressponds to index second to IG
        highest_information_gain= np.array([[0.0,0.0]])
        highest_information_gain = highest_information_gain.astype(float)
        highest_information_gain[0,1] = self.calculate_information_gain(attribute_nodes[0],
                                                                        label_one_value,
                                                                        label_two_value)

        #loop through and find the max information gain
        index_of_attribute = 0
            
        for attribute_node in attribute_nodes:
            #calculate information gain
            new_information_gain = self.calculate_information_gain(attribute_node,
                                                                   label_one_value, 
                                                                   label_two_value)

            #Compare it to the top value
            if new_information_gain > highest_information_gain[0,1]:
                
                #If it is greater, update the highest_information gain
                highest_information_gain[0,0] = index_of_attribute
                highest_information_gain[0,1] = new_information_gain
                
            #Update index to reflect new attribute
            index_of_attribute = index_of_attribute + 1
            
        return highest_information_gain
            
    
    #Assigns a information gain value to a attribute node
    def calculate_information_gain(self, attribute_node,class_label_one, class_label_two):
        
        #Calculating H(Y)
        class_entropy = self.calculate_entropy(attribute_node.data, class_label_one,
                                               class_label_two)
        
        #Variables needed to calculate the rest of the formula
        data_api = dataManipulator()
        
        #This gets us two values of the current attribute node
        attribute_label_one_value, attribute_label_two_value = data_api.attribute_values(attribute_node.data, 0)
        
        #This gets us two dataframes, one with all the data with attribute values one, the other with attribute values two
        dataframe_attribute_label_one, dataframe_attribute_label_two = data_api.divide_data_based_off_attribute_value(attribute_node.data,attribute_label_one_value, attribute_label_two_value,0)
        
        #These are used for finding P(X)
        label_one_total = np.count_nonzero(attribute_node.data[:,0] == attribute_label_one_value)
        label_two_total = np.count_nonzero(attribute_node.data[:,0] == attribute_label_two_value)
        
        #Lets sum up for total used in finding P(X)
        total_amount_of_data = label_one_total + label_two_total
        
        #Now calculating H(Y|X) 
        y_conditional_x_one = self.calculate_entropy(dataframe_attribute_label_one, 
                                                     class_label_one, class_label_two)
        y_conditional_x_two = self.calculate_entropy(dataframe_attribute_label_two, 
                                                     class_label_one, class_label_two )
        
        #Now calculating H(Y|X) = P(X) * H(Y|X) for X with value one        
        total_y_conditional_x = ((label_one_total/total_amount_of_data) * y_conditional_x_one) + ((label_two_total/total_amount_of_data) * y_conditional_x_two)
        
        #Finally, calculating IG is H(Y) - H(Y|X)
        information_gain = class_entropy - total_y_conditional_x
        
        return information_gain
        
if __name__ == '__main__':
    
    #Creating a user_input object to be used to get user input
    user_input_object = readInput()
    
    #Creating a classifer object to be used to classify data
    classifer_api = classifierOutputs()
    
    #creating a dictionary full of user input 
    user_input_values = user_input_object.user_input_dictionary()
    
    #Grabbing max_tree_depth for use later on
    max_depth = int(user_input_values["max_tree_depth"])
    
    #Reading in training data. Raw training data is for data without header removed
    data_api = dataManipulator()
    training_data, raw_training_data = data_api.import_data_to_array(user_input_object.training_data)
    
    #Reading in testing_data. Raw testing data is for data without header removed
    testing_data, raw_testing_data = data_api.import_data_to_array(user_input_object.testing_data)
    
    #Creating a list of attribute values in string
    attribute_values_strings = data_api.attributes_in_strings_list(raw_training_data)

    
    #Create list of attributes
    attribute_nodes=data_api.attribute_list(training_data)
    
    #Values of class data 
    class_label_one_value, class_label_two_value = data_api.attribute_values(training_data, -1)
    
    #Values of attributes data
    attribute_value_one, attribute_value_two = data_api.attribute_values(
                    training_data,0)
    
    #how many attributes we have
    attribute_total = len(attribute_nodes)
    
    # Setting up the tree
    root = Node(training_data)
    
    #setting inital depth
    root.depth = 0
    
    #tree API
    tree_grower = growTree()
    
    #inital print of distribution
    data_api.class_distribution_printer(training_data,class_label_one_value,
                                        class_label_two_value)
    
    #Creating the Tree. If max depth is 0, use majority vote classifier
    if max_depth == 0:
        root = tree_grower.majority_vote(training_data, root, class_label_one_value, 
                                         class_label_two_value, -1)
    #else, use the normal tree
    else:
        root=tree_grower.grow(root, max_depth, class_label_one_value,class_label_two_value, 
                             attribute_total, attribute_values_strings,attribute_value_one,
                             attribute_value_two)
    
    #Now test the tree on the training data
    
    #Creating a blank file for training labels file
    classifer_api.create_blank_initial_file(user_input_values["train_labels_file_name"])
    
    #Passing in all values of training data to be written to labels file
    for row in training_data:
        classifer_api.create_label_file_content(root, row, attribute_value_one, 
                                                attribute_value_two,
                                                user_input_values["train_labels_file_name"])
    
    #Creating a blank file for testing labels file
    classifer_api.create_blank_initial_file(user_input_values["test_labels_file_name"])
    
    #Passing in all values of test data to be written to a seperate labels file
    for row in testing_data:
        classifer_api.create_label_file_content(root, row, attribute_value_one, 
                                                attribute_value_two,
                                                user_input_values["test_labels_file_name"])
    
    #Now gathering error rates on training and testing predictions to output to metrics file
    training_data_classification_error = classifer_api.obtain_classification_error(training_data,user_input_values["train_labels_file_name"])
    testing_data_classification_error = classifer_api.obtain_classification_error(testing_data,user_input_values["test_labels_file_name"])
    
    classifer_api.print_metric_file(training_data_classification_error, 
                                              testing_data_classification_error,
                                              user_input_values["metrics_file_name"])
