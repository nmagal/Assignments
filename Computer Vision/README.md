# Computer Vision 
Please find below assignment descriptions for assignments given at CMU's Computer Vision course.  The following assignments were:

## Spatial Pyramid Matching for Scene Classification
In this assignment, I created a SPM scene classification system using a bag-of-words pyramid approach. The steps were as follows:
- Extracted filter responses on all training images using a Gaussian Filter, Laplacian of Gaussian Filter, derivative of Gaussian in the x direction, and derivative of Gaussian in the y direction at multiple scales.
- Created visual words by clustering on all of these filter responses to create a visual word dictionary.
- Given these visual words, assigned histograms of visual words to each picture.
- To run inference, labeled an image the label of the training image that has the smallest histogram intersection similarity difference.
- To make more robust, added a spatial prymaid.  This encourged the classification system to take into account the spatial structure of the image.
- For more information, please read the handout located in the project directory.

## Augmented Reality with Planar Homographies
In this assignment, I created an augmented reality video using homographies.  The steps were:
- Used the FAST detector with the BRIEF descriptor to identify correspondences between two seperate images.
- Used the resulting correspondences to solve for the homography between them using the constrained least squares approach.
- To further refine the homography, used RANSAC to deal with noisy correspondences. 
- Superimposed the warped image onto the destination image to create an AR image.
- For more information, please read the handout located in the project directory. 

## 3D Reconstruction from Stero Images
In this assignment I reconstructed the 3D scene of an object given two images seperated by a rotation and a translation. The steps were as follows:
- Estimated the fundemental matrix using the eightpoint algorithm.
- From the fundemental matrix, solved the essential matrix.
- Using the constraints from the epipolar line, found correspondences.
- From the correspondences, triangulated the object and created a 3D representation of the image.
- For more information, please read the handout located in the project directory.

## Neural Networks for Recognition
In this assignment, a neural network was created (using numpy and not pytorch/tensorflow) and trained to classify extracted text. Furthermore, an autoencoder was created.  The steps were as follows:
- Create a nn script, comprised of forward and backward pass functions.
- Train nn on training data.
- Extract text from images of handwritten text and keep extracted text in order by using clustering.
- Perform inference. 
- Create autoencoder and measure PSNR as an evaluation metric of performance. 
- For more information, please read the handout located in the project directory.
