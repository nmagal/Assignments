import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
import planarH as h
import pdb
import time

#This function will warp an image onto a destination image using a homography
def warpImage(opts):
    #Reading in images
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_ondesk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    
    #Getting correspondences
    matches, locs1, locs2 = matchPics(cv_ondesk, cv_cover, opts)
    locs1 = h.sort_locs(locs1)
    locs2 = h.sort_locs(locs2)
    locs1, locs2 = h.get_matches(matches, locs1, locs2)
        
    #Getting homography matrix
    homography_matrix, inlier_coordinates = h.computeH_ransac(locs1, locs2, opts)
    
    #We must warp our harry potter to be of the same size as computer vision textbook cover
    resize_dim = (cv_cover.shape[1], cv_cover.shape[0])
    hp_cover = cv2.resize(hp_cover, resize_dim)
    warp_size = cv_ondesk.shape[0:2]
    
    #Applying homography
    warped_hp = cv2.warpPerspective(src = hp_cover, M=homography_matrix, dsize = warp_size[::-1])
    
    #Superimposing warped image onto scene
    composite_photo = h.compositeH(warped_hp, cv_ondesk)
    
    #Displaying image
    cv2.imshow('sample image',composite_photo)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


