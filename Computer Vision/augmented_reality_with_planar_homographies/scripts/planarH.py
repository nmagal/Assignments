import numpy as np
import cv2
import pdb
from opts import get_opts
import random
import math
from matchPics import matchPics

#This function computes a homography between two seperate images
def computeH(x1, x2):
    
    A_components = []
    for matching_point_index in range(x1.shape[0]):
        component_1 = np.zeros(9)
        component_2 = np.zeros(9)
        
        component_1[0] = x2[matching_point_index, 0]
        component_1[1] = x2[matching_point_index, 1]
        component_1[2] = 1
        component_1[6] = -x1[matching_point_index,0] * x2[matching_point_index, 0]
        component_1[7] = -x1[matching_point_index,0] * x2[matching_point_index, 1]
        component_1[8] = -x1[matching_point_index,0]
        
        component_2[3] =  x2[matching_point_index, 0]
        component_2[4] = x2[matching_point_index, 1]
        component_2[5] = 1
        component_2[6] = -x1[matching_point_index, 1] * x2[matching_point_index, 0]
        component_2[7] = -x1[matching_point_index, 1] * x2[matching_point_index, 1]
        component_2[8] = -x1[matching_point_index, 1]
        
        A_components.append(component_1)
        A_components.append(component_2)
    
    #Constructing A matrix
    A_matrix = np.stack(A_components, axis = 0)
    at_a = np.dot(A_matrix.T,A_matrix)
    #Solving using eigendecomposition
    eig_vals, eig_vecs = np.linalg.eigh(at_a)  
    H2to1 = eig_vecs[:,0].reshape(3,3)
    
    #Normalizing homography 
    H2to1 = H2to1 / H2to1[-1, -1]
    
    return H2to1

# Solving our homography matrix with normalization
def computeH_norm(x1, x2):
     # Solving our homography matrix with normalization
     
     # Compute the centroid of the points
     x1_mean_x = np.mean(x1, axis =0)[0]
     x1_mean_y = np.mean(x1, axis =0)[1]
     
     x2_mean_x = np.mean(x2, axis =0)[0]
     x2_mean_y = np.mean(x2, axis =0)[1]
     
     # Similarity transform 1
     x1_scale = compute_scale(x1)
     T1 = create_similarity_matrix(x1_scale, x1_mean_x, x1_mean_y)
     
     # Similarity transform 2
     x2_scale = compute_scale(x2)
     T2 = create_similarity_matrix(x2_scale,x2_mean_x,x2_mean_y )
     
     # Shift the origin of the points to the centroid and Normalize the points so that the largest distance from the origin is equal to sqrt(2)
     x1_prime = np.dot(T1, x1.T).T
     x2_prime  = np.dot(T2, x2.T).T
     
     # Compute homography
     H2to1 = computeH(x1_prime, x2_prime)
      
     # Denormalization
     denomalized_H2to1 = np.linalg.inv(T1)@H2to1@T2
     denomalized_H2to1 = denomalized_H2to1/denomalized_H2to1[-1,-1]
     
     return denomalized_H2to1

# Compute the best fitting homography given a list of matching points using RANSAC
def computeH_ransac(locs1, locs2, opts):
    
    #the number of iterations to run RANSAC for
    max_iters = opts.max_iters  
    
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol
    
    #Converting into homogenous coordinates
    locs1, locs2 = add_homogenous_coordinates(locs1, locs2)
    
    homog_tracker = []
    for iteration in range(max_iters):
        
        #First computing Homography from random coordinates
        random_coordinates = random.sample(range(locs1.shape[0]), 4)
        H2to1 = computeH_norm(locs1[random_coordinates], locs2[random_coordinates])
        
        #Evaluating how many inliers we have 
        total_inliers = 0
        inlier_tracker = np.zeros(locs1.shape[0])
        
        #Test our homography on all different coordinates 
        for coordinate_index in range(locs1.shape[0]):
            locs2_cordinate = locs2[coordinate_index, :]
            warped_coordinate = np.dot(H2to1, locs2_cordinate)
            
            #If difference between warped cordinate and unwarped is below tolerance threshold, count as an inlier
            if np.linalg.norm(warped_coordinate/warped_coordinate[-1] - locs1[coordinate_index, :]) <= inlier_tol:
                
                total_inliers+=1  
                inlier_tracker[coordinate_index] = 1
        
        
        homog_tracker.append((total_inliers, H2to1, inlier_tracker ))
    
    #Choose our homography matrix as the one that had the most amount of inliers
    homog_tracker.sort(key=lambda x: x[0], reverse = True)
    return homog_tracker[0][1], homog_tracker[0][2]

#Switch x and y columns for better interpretability 
def sort_locs(locs):
    y_col = locs[:, 0]
    x_col = locs[:, 1]
    reformed_locs = np.stack((x_col, y_col), axis=1)
    return(reformed_locs)

#Return only corresponding points from locs1 and locs2
def get_matches(matches, locs1, locs2):
    matched_locs1 = locs1[matches[:, 0]]
    matched_locs2 = locs2[matches[:, 1]]   
    return(matched_locs1, matched_locs2)

#Create another dimension in order to use homogenous math 
def add_homogenous_coordinates(locs1, locs2):    
     one_colum = np.ones(locs1.shape[0])
     x1_homo = np.hstack((locs1, one_colum[:, None]))
     x2_homo = np.hstack((locs2, one_colum[:, None]))
     return(x1_homo, x2_homo)

#Computes the scale for linear transformations
def compute_scale(x):
    distance_from_origin = []
    
    for vector_index in range(x.shape[0]):
        distance_from_origin.append(np.linalg.norm(x[vector_index, :]))
    
    max_distance = max(distance_from_origin)
    scale = math.sqrt(2) / max_distance 
    return scale

#Creates matrix to solve for linear transformations
def create_similarity_matrix(scale, x_mean, y_mean):
     sim_matrix = np.array([[scale, 0, -scale*x_mean],
                    [0, scale, -scale*y_mean],
                    [0,0,1]])    
     return(sim_matrix)

#Create a composite image after warping the template image on top of the image using the homography
def compositeH(template, img):
     mask_indexes = np.nonzero(template[:,:, 0])
     img[mask_indexes] = 0  
     combined_photo = img + template
     
     return combined_photo
 
 
