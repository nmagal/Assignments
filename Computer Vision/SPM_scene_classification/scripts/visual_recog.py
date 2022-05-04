import os
import math
import multiprocessing
from os.path import join
from copy import copy
import numpy as np
from PIL import Image
import visual_words
from visual_words import get_visual_words
import pdb 
from itertools import repeat
from sklearn.metrics import confusion_matrix


def get_feature_from_wordmap(wordmap, opts):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """
    #Reading in variables 
    K = opts.K
    
    #Creating a histogram of our wordmap
    wordmap_flat = wordmap.flatten()
    hist = np.histogram(wordmap_flat, bins = K)
    
    #Normalizing our histogram
    total_sum = sum(hist[0])
    new_hist_values = [x/total_sum for x in hist[0]]
    hist = (new_hist_values, hist[1])
    
    return(new_hist_values)


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    """
    #Reading in variables
    K = opts.K
    L = opts.L
    wordmap_h, wordmap_w = wordmap.shape
    
    #Creating weighting scheme for different levels on the pyramid 
    first_two_weights = 2**(-L)
    remaining_weights = [ 2**(x - L -1) for x in range(2, L+1)]
    combo_weights=[first_two_weights, first_two_weights]
    combo_weights.extend(remaining_weights)
    assert sum(combo_weights) == 1, "Not a valid weight scheme, weights add up to: " +str(sum(combo_weights))
    
    #Breaking down the image into multiple spatial pyramid levels composed of cells 
    cells_all_levels_histograms = []
    for level in range(L + 1):
        num_cells = 2**level * 2**level
        cells_individual_level = []
        cell_resolution = (math.ceil(wordmap_h/2**level), math.ceil(wordmap_w/2**level))
        
        #Creating cells 
        for h_index in range(2**level):
            for w_index in range(2**level):
                h0 = h_index * cell_resolution[0]
                h1 = (h_index+1)*cell_resolution[0]
                w0 = w_index * cell_resolution[1]
                w1 = (w_index+ 1)* cell_resolution[1]
                new_cell = wordmap[h0:h1, w0:w1]
                cells_individual_level.append(new_cell)
                
        assert num_cells == len(cells_individual_level), "not the correct number of cells"
        
        #Now getting histograms from cells
        histograms = []
        for cell in cells_individual_level:
            #Getting the histogram of the cell 
            unweighted_histogram = get_feature_from_wordmap(cell, opts)
            #applying weighting scheme to cell histogram
            weighted_histogram = np.asarray(unweighted_histogram) * combo_weights[level]/(len(cells_individual_level))
            histograms.append(weighted_histogram)
        
        #Append all cells of layer level to list
        cells_all_levels_histograms.append(histograms)
    
    #Reshaping histograms to be one giant histogram 
    final_hist = []
    for hist_list in cells_all_levels_histograms:
        for individual_histograms in hist_list:
            final_hist.append(individual_histograms)
    final_hist = np.asanyarray(final_hist).reshape(1,-1)
    projected_hist_length = (K*(4**(L+1)-1))/3
    assert projected_hist_length == final_hist.shape[1], " projected and final_hist length do not match up"
 
    return(final_hist)
    

def get_image_feature(opts, train_files, dictionary, concat_output = True):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """
    
   #Getting filter responses
    with multiprocessing.Pool() as pool:
        visual_words = pool.starmap(get_visual_words,zip(repeat(opts), train_files, repeat(dictionary), repeat(True)))
 
    #Getting all training histograms from the spatial prymaid, will be a N x (K*(4**(L+1)-1))/3
    with multiprocessing.Pool() as pool:
        training_histograms = pool.starmap(get_feature_from_wordmap_SPM, zip(repeat(opts), visual_words))
    
    if concat_output:
        training_histograms = np.concatenate(training_histograms, axis = 0)

    return(training_histograms)

def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """
    
    #Reading in variables
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "../output/dictionary.npy")) 
    training_histograms = get_image_feature(opts, train_files, dictionary)

    #Saving the learned system
    np.savez_compressed(join(out_dir, '../output/trained_system.npz'),
        features=training_histograms,
         labels=train_labels,
         dictionary=dictionary,
         SPM_layer_num=SPM_layer_num,
     )
    

def distance_to_set(word_hist, histograms_training, labels):
    
    assert word_hist.shape[1] == histograms_training.shape[1], " Histogram shapes are not equal: word hist and histograms comparision are : "+str(word_hist.shape[1]) + str(histograms_training.shape[1])
    
    #finding the image that has the closest histogram to inference image using histogram intersection similarity 
    minimum = np.minimum(word_hist, histograms_training)
    total_sum = np.sum(minimum, axis = 1, keepdims = True )  
    distances = 1 - total_sum
    index_of_label = np.argmin(distances, axis=0)
    label = labels[index_of_label]
    return(label)


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    #Reading in variables 
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    trained_system = np.load(join(out_dir, "../output/trained_system.npz"))
    dictionary = trained_system["dictionary"]

    #Using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]
    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    training_data = trained_system['features'] 
    training_labels = trained_system['labels']

    #Loads image --> gets filter responses --> maps to words --> gets SMP
    test_features = get_image_feature(opts, test_files, dictionary, False)
    
    #Getting predictions
    with multiprocessing.Pool() as pool:
        predicted_labels = pool.starmap(distance_to_set, zip(test_features,repeat(training_data), repeat(training_labels)))    
    predicted_labels = np.concatenate(predicted_labels, axis=0)
    
    #Obtaining quantitative results
    results = confusion_matrix(test_labels, predicted_labels, labels= [0,1,2,3,4,5,6,7])
    acc_per_label = results.diagonal()/results.sum(axis = 1)
    total_acc = sum(results.diagonal())/len(test_labels)
    return(results, total_acc)

