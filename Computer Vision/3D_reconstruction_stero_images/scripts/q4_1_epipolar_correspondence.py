import numpy as np
import matplotlib.pyplot as plt
from helper import _epipoles
from q2_1_eightpoint import eightpoint
from scipy.ndimage import gaussian_filter
import pdb

# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]
        
        #This returns x and y coordinates
        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        #Find the corresponding line equation based off of point in im1 and F matrix 
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', MarkerSize=8, linewidth=2)
        plt.draw()

#Get's the coordinates of epipolar line given a point from im1 and F matrix
def get_line_coordinates(x1,x2, im2, F):
    
    im_height, im_width, _ = im2.shape
    point_to_match = np.array([x1,x2, 1])
    
    #Now finding the corresponding line
    l = F.dot(point_to_match)
    s = np.sqrt(l[0]**2+l[1]**2)
    l = l/s
    y_points = [*range(int(np.round(x2))-50,int(np.round(x2))+50,1)]
    x_points =[]
    
    for y_point in y_points:
        if l[0] != 0:
            x_points.append(int(np.round(-(l[1] * y_point + l[2])/l[0])))
        else:
            x_points.append(int(np.round(-(l[0] * y_point + l[2])/l[1])))
            
    line_coordinates = list(zip(y_points, x_points))
    return(line_coordinates)

#Computes the distance between canidiate corresponding points and gives us the most similar one
def similarity_measure(x1,y1, im1, im2, potential_coordinates, window_size, pixel_similarity = True):
    
    #Measuring the distance between potential points
    distance = []
    for coordinate_pair in potential_coordinates:
        #pixel similarity se
        if pixel_similarity == True or coordinate_pair[0]-window_size <0 or coordinate_pair[1]-window_size <0 or coordinate_pair[0]+window_size+1>480 or coordinate_pair[1]+window_size+1 >640:
            #Matching im1 pixel to pixel in im2
            im1_pixel_value = im1[y1, x1]
            
            #Canidate corresponding point
            im2_pixel_value = im2[coordinate_pair[0], (coordinate_pair[1])]
            
            #Storing the distance betwen im1 pixel and canidate im2 pixel
            distance.append(np.linalg.norm(im1_pixel_value-im2_pixel_value))
            
        else:
            #Creating grid cordinates for im1
            x_min_im = y1[0]-window_size
            x_max_im = y1[0]+window_size+1
            
            y_min_im= x1[0]-window_size
            y_max_im = x1[0]+window_size+1
            
            #The window that stores our im1 patch 
            im1_window_value = im1[x_min_im:x_max_im, y_min_im:y_max_im]
            
            #Creating grid cordinates for im2
            x_min_con = coordinate_pair[0]-window_size
            x_max_con = coordinate_pair[0]+window_size+1
            
            y_min_con = coordinate_pair[1]-window_size
            y_max_con =coordinate_pair[1]+window_size+1
            
            #The window that stores our im2 patch 
            im2_window_value = im2[x_min_con:x_max_con, y_min_con:y_max_con]
            
            #Applying Gaussian smoothing to reduce noise
            im1_window_value = gaussian_filter(im1_window_value, sigma=4)
            im2_window_value = gaussian_filter(im2_window_value, sigma=4) 
            
            distance.append(np.linalg.norm(im1_window_value-im2_window_value)) 

    #The lowest distance will be the corresponding point/window
    matching_point = np.argmin(distance)
    return(potential_coordinates[matching_point])
        
def epipolarCorrespondence(im1, im2, F, x1, y1, pixel_similarity = True):
    #Finding our epipolar line
    line_coordinates = get_line_coordinates(x1, y1, im2, F)
    
    #Getting our matching point
    matching_coordinates = similarity_measure(x1, y1, im1, im2, line_coordinates, 6, pixel_similarity)#4

    #Note returns in Y(col), X(row) format for matplotlib
    return(matching_coordinates[1],matching_coordinates[0])


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
if __name__ == "__main__":
    
    #Reading in variables
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    
    #Getting our fundemental matrix 
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
   
    #Creating correspondences using epipolar geometry
    epipolarMatchGUI(im1, im2, F)
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    print("Distance is",str(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) ))
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)