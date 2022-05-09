import numpy as np
import matplotlib.pyplot as plt
from helper import camera2
from q2_1_eightpoint import eightpoint, _singularize
import pdb

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    #Finding the Esential matrix
    E = K2.T @ F @ K1
    E = _singularize_essential(E)
    E = E/E[-1,-1]
    
    #Saving our Essential Matrix
    np.savez('../output/q3_1.npz',E)
    return(E)

#Singularizes essential matrix to be of format of the definition of essential matrices
def _singularize_essential(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    av = (S[0] + S[1])/2
    S[0] = av
    S[1] = av
    F = U.dot(np.diag(S).dot(V))
    return F
    
if __name__ == "__main__":
    #Reading in variables
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    
    #Calculate the Fundemental matrix, use this to calculate the essential matrix
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    E = essentialMatrix(F, K1, K2)

    # Simple Tests to verify your implementation:
    assert(E[2, 2] == 1)
    assert(np.linalg.matrix_rank(E) == 2)