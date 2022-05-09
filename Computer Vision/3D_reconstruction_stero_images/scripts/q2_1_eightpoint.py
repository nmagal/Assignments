import numpy as np
import matplotlib.pyplot as plt
import os 
from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize
import pdb


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    
    #Converting correspondences to homogenous coordinates  
    pts1 = toHomogenous(pts1)
    pts2 = toHomogenous(pts2)
     
    #Scale Matrix
    T = np.array([[1/M, 0, 0],
                 [0,1/M, 0],
                 [0,0,1]])

    #Scaling correspondences
    pts1 = (T @ pts1.T).T
    pts2 = (T @ pts2.T).T
    
    #Setting up eight point algorithm
    
    A_list =[]
    for index in range(pts2.shape[0]):
        A_list.append(eight_point_equation(pts2[index],pts1[index]))
    
    A = np.stack(A_list, axis =0)  
    U, S, V_transpose = np.linalg.svd(A)
    
    #Solving for our fundamental matrix
    F = V_transpose[-1].reshape(3,3)
    
    #Enforcing the singularity and refining 
    F = _singularize(F)
    F = refineF(F, pts1[:, :2], pts2[:, :2])

    #Unscaling 
    F = T.T@F@T
    F = F/F[-1,-1]
    
    #Saving F and M 
    np.savez('../output/q2_1.npz', f_matrix=F, m_scale = T)
    return(F)
    
    
#Given two matching points, return one equation from the 8_point_equation
def eight_point_equation(cor_0, cor_1):
    
    x = cor_0[0]
    x_prime = cor_1[0]    
    
    y = cor_0[1]
    y_prime = cor_1[1]
    
    equation = np.array([x*x_prime, x*y_prime, x, y*x_prime, y*y_prime, y, x_prime, y_prime, 1])
    
    return(equation)
    
    
if __name__ == "__main__":

    #Reading in variables
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    #Creating fundemental matrix
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # Simple Tests to verify your implementation:    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    displayEpipolarF(im1, im2, F)
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1), str(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)))
    
    