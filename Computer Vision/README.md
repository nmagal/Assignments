# Computer Vision 
Please find below assignment descriptions for assignments given at CMU's Computer Vision course.  The following assignments were:

## Spatial Pyramid Matching for Scene Classification
In this assignment, we created a SPM scene classification system using a bag-of-words pyramid approach. The steps were as follows:
- Extract filter responses on all training images using a Gaussian Filter, Laplacian of Gaussian Filter, derivative of Gaussian in the x direction, and derivative of Gaussian in the y direction at multiple scales.
- Create visual words by clustering on all of these filter responses to create a visual word dictionary.
- Given these visual words, assign histograms of visual words to each picture.
- To run inference, label an image the label of the training image that has the smallest histogram intersection similarity difference.
- To make more robust, add a spatial prymaid.  This encourges the classification system to take into account the spatial structure of the image.
- For more information, please read the handout located in the project directory.

## Augmented Reality with Planar Homographies
In this assignment, we created an augmented reality video using homographies.  The steps were:
- Use the FAST detector with the BRIEF descriptor to identify correspondences between two seperate images.
- Use the resulting correspondences to solve for the homography between them using the constrained least squares approach.
- To further refine the homography, use RANSAC to deal with noisy correspondences. 
- Superimpose the warped image onto the destination image to create an AR image.
- For more information, please read the handout located in the project directory. 
