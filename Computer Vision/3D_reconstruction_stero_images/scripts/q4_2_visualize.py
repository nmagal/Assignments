import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence
import pdb

'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):
    
    #Reading in varaibles 
    temple_coord_x = temple_pts1['x1']
    temple_coord_y = temple_pts1['y1']
    
    #Finding corresponding points for temple_coord
    corresponding_points = []
    for i in range(temple_coord_x.shape[0]):
        corresponding_points.append(epipolarCorrespondence(im1, im2, F,temple_coord_x[i], temple_coord_y[i], pixel_similarity=False))
    
    #We need to put the points into the form of N x 2 numpy array
    temple_coords = [np.array([temple_coord_x[index][0],temple_coord_y[index][0]]) for index in range(temple_coord_x.shape[0])]
    temple_coords = np.stack(temple_coords, axis=0)
    
    corresponding_points = [np.array([coordinates[0], coordinates[1]]) for coordinates in corresponding_points]
    corresponding_points = np.stack(corresponding_points, axis =0)
    
    #Now finding the 3D points 
    M2, C2, P = findM2(F,temple_coords, corresponding_points,intrinsics)
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)

    #Saving output
    np.savez('../output/q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2 )
    
    #Graphing 
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(3,5)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.scatter(P[:, 0],P[:, 1], P[:, 2])

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    #Reading in variables
    temple_coords_path = np.load('../data/templeCoords.npz')
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    
    #Create fundemental matrix
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    
    #Compute and render 3D points
    compute3D_pts(temple_coords_path, intrinsics, F, im1, im2)

