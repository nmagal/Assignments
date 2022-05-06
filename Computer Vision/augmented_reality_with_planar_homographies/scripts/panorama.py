#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:56:09 2022

@author: nicholasmagal
"""
import cv2 
from matchPics import matchPics
from opts import get_opts
import planarH as h
import pdb
import numpy as np
from PIL import Image

#Create a simple panoramic image using homographies 
def pano():
    #Reading in images
    pano_left = cv2.imread('../data/left.jpeg')
    pano_right = cv2.imread('../data/right.jpeg')
    opts = get_opts()
    
    #Getting correspondences
    matches, locs1, locs2 = matchPics(pano_left, pano_right, opts)
    locs1 = h.sort_locs(locs1)
    locs2 = h.sort_locs(locs2)
    locs1, locs2 = h.get_matches(matches, locs1, locs2)
    
    #Full panoramic size
    warp_size = np.asarray(pano_left.shape[0:2]) + np.asarray(pano_right.shape[0:2])
    
    #Getting homography and applying it
    homography_matrix, inlier_coordinates = h.computeH_ransac(locs1, locs2, opts)
    warped_photo = cv2.warpPerspective(src = pano_right, M=homography_matrix, dsize = warp_size[::-1])
    
    #Adding in panoleft to warped pano right
    warped_photo[:pano_left.shape[0], :pano_left.shape[1]] =  pano_left
    
    #Cropping black space out
    warped_photo = warped_photo[:1200, :-800]
    
    #Displaying panoramic image
    cv2.imshow('pano',warped_photo)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pano()
    