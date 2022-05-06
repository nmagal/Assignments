import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import pdb


#This function will display correspondences between two different images
def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """
    
    #Get our interest points and matched corresponding interest points
    matches, locs1, locs2 = matchPics(image1, image2, opts)
    
    #Display matched features
    plotMatches(image1, image2, matches, locs1, locs2)

if __name__ == "__main__":
    
    #Getting variables
    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    #Function to display correspondences 
    displayMatched(opts, image1, image2)

