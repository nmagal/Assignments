import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
import pdb

#This function finds correspondences between two images
def matchPics(I1, I2, opts):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """
    
    #ratio for BRIEF feature descriptor
    ratio = opts.ratio  
    #threshold for corner detection using FAST feature detector
    sigma = opts.sigma  
    
    #Convert images to grayscale
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)/255
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)/255

    #Detect interest points in both images
    I1_interest_points = corner_detection(I1, sigma)
    I2_interest_points = corner_detection(I2, sigma)
    
    #Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, I1_interest_points)
    desc2, locs2 = computeBrief(I2, I2_interest_points)
    
    #Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    #matches contain row index to locs1 and locs2
    return matches, locs1, locs2
