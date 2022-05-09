import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.restoration import denoise_wavelet
from skimage.filters import try_all_threshold
from skimage.morphology import closing, square, binary_dilation,binary_erosion
import skimage.segmentation
from skimage.segmentation import clear_border


# takes a color image and returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    
    #First denoise image  
    image = denoise_wavelet(image, channel_axis=-1, convert2ycbcr=True,
                                rescale_sigma=True)
    
    #Convert to greyscale
    image = rgb2gray(image)
    
    #Create a threshold of foreground vs background images
    thresh = skimage.filters.threshold_otsu(image)
    
    #Turning image into a binary image
    black_white= image
    black_white[image < thresh] = 0
    black_white[image >= thresh] = 1
    
    #Applying dialation/erosion
    filter_ = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])
    
    bw = closing(image < thresh, filter_)
    
    #Adding further dialation
    num_dialtions =5
    for i in range(num_dialtions):
        black_white = binary_erosion(black_white,filter_)
    
    #Removing artificats on image border
    cleared = clear_border(bw)
    
    #labeling
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    
    #Create bounding boxes
    for region in regionprops(label_image):
    # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minr, minc, maxr, maxc])
            
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    
    return bboxes, black_white