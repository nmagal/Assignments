import os
import multiprocessing
from os.path import join, isfile
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans
import pdb
from itertools import repeat


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """
    #If  statements to catch if channels are not in 3 channels format. First case is for greyscale second is if there are 4 channels
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=2)
    if img.shape[2]== 4: 
        img = img[:,:, 0:3]
    
    #Ensure that our image is normalized 
    assert img[0,0,0] <= 1, "Image is not normalized, pixel value is: "+str(img[0,0,0])

    #Converting image into rgb2lab
    img = skimage.color.rgb2lab(img)
    
    #Getting filter scales 
    filter_scales = opts.filter_scales
    
    filter_response = []
    #Applying each filter scale on each image channel
    for scale in filter_scales:
        guassian_filters = []
        gaussian_laplace_filters = []
        gaussian_x_der_filters = []
        gaussian_y_der_filters = []
        
        for channel in range(3):
            guassian_filters.append(scipy.ndimage.gaussian_filter(img[:,:,channel], (scale,scale), mode = 'constant'))
            gaussian_laplace_filters.append(scipy.ndimage.gaussian_laplace(img[:,:,channel], (scale,scale), mode ='constant'))
            gaussian_x_der_filters.append(scipy.ndimage.gaussian_filter(img[:,:,channel], (scale,scale), order=(1,0), mode='constant'))
            gaussian_y_der_filters.append(scipy.ndimage.gaussian_filter(img[:,:,channel], (scale,scale), order=(0,1), mode ='constant'))
            
        filter_response.append(np.stack(guassian_filters, axis=2))
        filter_response.append(np.stack(gaussian_laplace_filters, axis=2))
        filter_response.append(np.stack(gaussian_x_der_filters, axis=2))
        filter_response.append(np.stack(gaussian_y_der_filters, axis=2))
    
    #Turning our filter responses into the required H,W,3F shape
    filter_response = np.concatenate(filter_response, axis=2)
    assert img.shape[0:2] == filter_response.shape[0:2], "Filtered shapes are not equal"
    
    return(filter_response)
    

def compute_dictionary_one_image(image_path, opts):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    #Opening and preprocessing image 
    image_path = join(opts.data_dir, image_path)
    img = Image.open(image_path)
    img = np.array(img).astype(np.float32) / 255 
    
    #Getting filter responses 
    filter_response = extract_filter_responses(opts, img)
    x_shape, y_shape, channel_shape = filter_response.shape
    
    #Randomly sampling coordinates to create dictionary from
    random_x_coordinates = np.random.randint(0, high= x_shape, size = opts.alpha)
    random_y_coordinates = np.random.randint(0, high= y_shape, size = opts.alpha)
    random_responses = filter_response[random_x_coordinates,random_y_coordinates,:]
    
    #Saving random responses
    np.save(join(opts.data_dir,'temp_files', image_path.split("/")[-1]), random_responses)
    
    return(random_responses)


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """
    #variables set in opts.py file
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    
    #Reading in training data
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()

    #If temp_files doesn't exist, create it for saving output
    if not os.path.exists(join(data_dir,"temp_files")):
        os.mkdir(join(data_dir,"temp_files"))
    
    #Extracting Filter Responses over all training data
    with multiprocessing.Pool() as pool:
        filter_responses = pool.starmap(compute_dictionary_one_image,zip(train_files, repeat(opts)))
    
    #Reshaping filter responses for KMeans 
    f_r_sep = []
    for image in filter_responses:
        for fv_index in range(image.shape[0]):
            f_r_sep.append(image[fv_index,:])
    
    #Clustering
    kmeans = KMeans(n_clusters = K, n_jobs = n_worker).fit(f_r_sep)
    dictionary = kmeans.cluster_centers_
    
    #Saving our dictionary 
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary, read_in_image = False):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """
    
    #If we need to read in an image, do so here
    if read_in_image == True:
        img = join(opts.data_dir, img)
        img = Image.open(img)
        img = np.array(img).astype(np.float32)/255
    
    #Extract filters and reshape to pixels x channels 
    filter_response_img = extract_filter_responses(opts, img)
    flattened_filter_response = filter_response_img.reshape(filter_response_img.shape[0]*filter_response_img.shape[1],filter_response_img.shape[2])
    
    #Compute cartesian distance between pixels and visual words, and take argmin to assign visual word to pixels 
    euclidean_distanes = scipy.spatial.distance.cdist(flattened_filter_response, dictionary)
    img_visual_words = np.argmin(euclidean_distanes,axis =1).reshape(img.shape[0],img.shape[1])

    return(img_visual_words)