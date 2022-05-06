import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import pdb
import scipy
import matplotlib.pyplot as plt
from helper import plotMatches

#This function tests how our interest point detector and descriptor work under rotation
def rotTest(opts):

    #Reading the image
    image = cv2.imread('../data/cv_cover.jpg')
    
    #Compute how many corresponding points are found for each different rotation angle
    histogram = []
    rotation_values = [0, 50, 180]
    for i in rotation_values:
        # Rotate Image
        rot_image = scipy.ndimage.rotate(image, angle = (i))

        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(image, rot_image, opts)
        plotMatches(image,image,matches,locs1,locs2)
        
        #Append to histogram total matches of current rotation angle
        histogram.append(matches.shape[0])

    # Display histogram
    fig = plt.figure()
    str_rot_values = [str(rot) for rot in rotation_values]
    plt.bar(str_rot_values, histogram, width = .4)
    plt.xlabel("Degrees Rotated")
    plt.ylabel("Matches")
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
