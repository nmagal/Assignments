# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 27, 2022
# ##################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage import io
from skimage.color import rgb2xyz, xyz2rgb, rgb2gray
from utils import plotSurface, integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    #These are our surface normals
    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0]/2) * pxSize*1.e-4
    Y = (Y - res[1]/2) * pxSize*1.e-4
    Z = np.sqrt(rad**2+0j-X**2-Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)

    image = np.zeros((2160,3840))
    
    #Given an lambertian surface, take the dot product of the normals and of the light to get image radience 
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            image[row, col] = np.array([X[row, col], Y[row, col], Z[row, col]]) @ light
    
    return image

def loadData(path = "../data/", debug = False):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    
    #Reading in image and converting to xyz image space
    image_folder = []
    for i in range(1, 8):
        image_name = path +'input_'+str(i)+'.tif'
        image = imread(image_name)
        image = rgb2xyz(image)
        s = image.shape[0:2]
        image_folder.append(image)
    
    #Extracting the luminance channel Y and vectorizing images
    im_fol_lum_flat = []
    
    for image in image_folder:
        im_fol_lum_flat.append(image[:,:, 1].flatten())
    

    I = np.stack(im_fol_lum_flat, axis = 0)
    
    #Getting image lighting directions
    L = np.load(path+"sources.npy")

    return I, L, s

def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
  
    #Now solving
    B = np.linalg.inv(L.T @ L) @ L.T @ I
    # Your code here
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    
    albedos = np.linalg.norm(B, axis = 0) 
    normals = B/albedos
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    
    #Reshape albedos to image size
    albedoIm = albedos.reshape(s)
    
    #Reshape normals to image size
    normal_channels = []
    for i in range(3):
        normal_channels.append(normals[i, :].reshape(s))
    normalIm = np.stack(normal_channels, axis=2)

    #Normalize pixel values between 0 and 1
    normalIm = (normalIm- np.min(normalIm))/(np.max(normalIm) - np.min(normalIm))
    
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    
    #Create normal image
    normal_channels = []
    for i in range(3):
        normal_channels.append(normals[i, :].reshape(s))
    
    normals = np.stack(normal_channels, axis=2)
    
    #Calculate dx and dy
    dx = -normals[:, :, 0]/normals[:, :, -1] 
    dy = -normals[:, :, 1]/normals[:, :, -1] 
    
    #integrate with frankot algorithm
    surface= integrateFrankot(dx, dy)
    
    return surface

if __name__ == '__main__':
    
    # Part 1(b) - Rendering n-dot-l lighting
    radius = 0.75 # cm
    center = np.asarray([0, 0, 0]) # cm
    pxSize = 7 # um
    res = (3840, 2160)
    
    #Rendering sphere given light direction
    light = np.asarray([1, 1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)

    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('../output/1b-a.png', image, cmap = 'gray')

    #Rendering sphere given light direction
    light = np.asarray([1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('../output/1b-b.png', image, cmap = 'gray')
    
    #Rendering sphere given light direction
    light = np.asarray([-1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('../output/1b-c.png', image, cmap = 'gray')
    
    # Part 1(c) - Loading data (Image matrix, Lighting matrix, image size)
    I, L, s = loadData('../data/')

    # Part 1(d) - Inspecting rank of I matrix
    U, S, V_transpose = np.linalg.svd(I, full_matrices=False)
    print(S)
    
    # Part 1(e) - Estimating pseudonormals
    B = estimatePseudonormalsCalibrated(I, L)
    
    # Part 1(f) - Estimate albedos and normals 
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    plt.figure()
    plt.imshow(albedoIm.clip(0, .5), cmap = 'gray')
    plt.imsave('../output/1f-a.png', albedoIm.clip(0, .5), cmap = 'gray')
    
    plt.figure()
    plt.imshow(normalIm , cmap = 'rainbow')
    plt.imsave('../output/1f-b.png', normalIm, cmap = 'rainbow')

    # Part 1(i) - Integrate normals for 3d reconstruction 
    surface = estimateShape(normals, s)
    plotSurface(surface)