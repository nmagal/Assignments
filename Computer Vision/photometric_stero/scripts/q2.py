# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 27, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """
    
    #Using a rank 3 approximation to solve for L and B 
    U, S, V_transpose = np.linalg.svd(I, full_matrices=False)
    Sigma = np.sqrt(np.diag(S[0:3]))
    
    #B should be of size 3 x P 
    B = U[:, :3]
    
    #L should be of size 3 x 7 shape 
    L = Sigma @ V_transpose.T[:, :3 ].T

    return B, L

def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    #Bas Relief Transformation
    G = np.array([[1,0,0],
                  [0,1,0],
                  [mu, nu, lam]])
    B = np.linalg.inv(G).T @ B
    
    #Calculating albedo and normals
    transformed_albedos, transformed_normals = estimateAlbedosNormals(B)
    
    #Calculating/plotting surface 
    surface = estimateShape(transformed_normals, s)
    plotSurface(surface)

if __name__ == "__main__":

    # Part 2(b) - estimating pseudonormals uncalibrated
    I, L_ground, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I.T)
    
    #Calculating albedos and normals from pseudonormals
    albedos, normals = estimateAlbedosNormals(B.T)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    #Albedos visual
    plt.figure()
    plt.imshow(albedoIm , cmap = 'gray')
    plt.imsave('../output/2b.png', albedoIm, cmap = 'gray')
    
    #Normals visual
    plt.figure()
    plt.imshow(normalIm , cmap = 'rainbow')
    plt.imsave('../output/2b0.png', normalIm, cmap = 'rainbow')

    # Part 2 (d) Integrate using Frankot-Chellappa
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (e) Integrate using Frankot-Chellappa and enforcing integrability
    
    #Enforce integrability
    transformed_B = enforceIntegrability(B.T, s)
    
    #Get the normals and albedos
    transformed_albedos, transformed_normals = estimateAlbedosNormals(transformed_B)
    
    #Plot the surface
    surface = estimateShape(transformed_normals, s)
    plotSurface(surface)
    
    # Part 2 (f) Bas relief transformation visualization 
    plotBasRelief(transformed_B,1,15,1)
