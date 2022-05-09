import numpy as np
import matplotlib.pyplot as plt
from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix
import pdb

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):

    #Create A 
    a_list = []
    for index in range(pts1.shape[0]):
        a_list.append(a_creator(pts1[index], pts2[index], C1, C2))
    
    #Now solving for the 3d points
    three_d_points = []
    for index in range(len(a_list)):
        U,S,V_transpose = np.linalg.svd(a_list[index])
        point = V_transpose[-1].reshape(1,-1)
        three_d_points.append(point)
    
    #Now solving for the reprojection error
    reprojection_error = 0

    for index in range(len(three_d_points)):
        #Calculating for camera 1
        reprojection_error+= reprojection_calc(pts1[index], three_d_points[index], C1)
        
        #Calculating for camera 2
        reprojection_error+= reprojection_calc(pts2[index], three_d_points[index], C2)
    
    #Reformating and normalizing 3d coordinates
    three_d_points = np.concatenate(three_d_points, axis =0)
    three_d_points = three_d_points/three_d_points[:,3].reshape(-1,1)
    three_d_points = three_d_points[:, :-1]
    
    return(three_d_points, reprojection_error)
    
#Used for triangulation, A matrix 
def a_creator(cor1,cor2, C1, C2):
    
    x = cor1[0]
    y = cor1[1]
    
    x_prime = cor2[0]
    y_prime = cor2[1]
    
    one = y*C1[-1] - C1[1]
    two = C1[0] - x*C1[-1]
    three = y_prime*C2[-1] - C2[1]
    four = C2[0] - x_prime*C2[-1]
    
    A = np.stack([one, two, three, four], axis=0)
       
    return(A)

#Calculates the reprojection error
def reprojection_calc(twod_coordinate, threed_coordinate, camera_matrix):
    
    reprojection = camera_matrix @ threed_coordinate.T
    reprojection = reprojection/reprojection[-1]
    reprojection = reprojection[:-1]
    error = np.linalg.norm(twod_coordinate.reshape(2,1) - reprojection)
    return(error)

def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''
    
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    M2 = camera2(E) 
    C1 = K1.dot(M1)
    
    #triangulating coordinates using different M matrices 
    errors = []
    points = []
    for i in range(M2.shape[2]):
        
        C2 = K2.dot(M2[:,:,i])
        coord, error = triangulate(C1, pts1, C2, pts2)
        errors.append(error)
        points.append(coord)
    
    #The M2 matrice should not be projecting with negative Z values, so find the one that only has positive z values
    neg_z = []
    for i in range(4):
        under_0 = points[i][:, -1] < 0
        under_0 = True in under_0
        neg_z.append(under_0)
    
    possible_indexes = [i for i, x in enumerate(neg_z) if x==False]
    assert(len(possible_indexes) == 1)
    
    #Choosing the correct variables based off of previous M2 constraint
    M2 = M2[:,:,possible_indexes[0]]
    P = points[possible_indexes[0]]
    C2 = K2.dot(M2)
    
    #Saving these outputs
    np.savez('../output/q3_3.npz', M2=M2, P=P, C2=C2)
    return M2, C2, P
    
if __name__ == "__main__":
    
    #Reading in variables
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    #Caluclating fundemental matrix
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    
    #Calculating camera projection matrix , C2(intrinsics @ extrinsics), and the reconstructed 3d points
    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert(err < 500), str(err)