import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
from sklearn.cluster import MeanShift
from nn import *
from q4 import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#Extracting and classifying text
for img in os.listdir('../images'):
    
    #Getting image
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images','04_deep.jpg'), plugin ='matplotlib'))
    
    #Getting all the bounding boxes of text from image
    bboxes, bw = findLetters(im1)
    
    #This visualizes our image with bounding boxes
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    # find the rows using clustering in order to classify text from left to right top to bottem
    y_coordinates = np.asarray([top_y for top_y, _, _, _, in bboxes]).reshape(-1,1)
    clustering = MeanShift(bandwidth=70).fit(y_coordinates.reshape(-1,1))
    cluster_indexes = clustering.labels_
    
    #getting unique amount of rows
    unique_clusters = (set(cluster_indexes))
    
    organized_letters = []
    #Below we divide into rows, and then sort the rows by their columns, then sort the rows
    for cluster_row in unique_clusters:
        indexes = np.where(cluster_indexes == cluster_row)
        row = [bboxes[i] for i in indexes[0]]
        row.sort(key = lambda x: x[1])
        organized_letters.append(row)
    
    #Now sorting by row
    organized_letters.sort(key = lambda x: x[0][0])
    
    #Finally combining list
    bboxes = sum(organized_letters, [])

    #Cropping and modifying extracted text to match training data
    cropped_letters = []
    for bb in (bboxes):
        cropped_letter = bw[bb[0]:bb[2]+1, bb[1]:bb[3]+1]
        cropped_letter = np.pad(cropped_letter, pad_width=40, constant_values = 1)
        
        #Centering
        center_cols = np.ones((cropped_letter.shape[0],10))
        cropped_letter = np.hstack((center_cols, cropped_letter))

        cropped_letter = skimage.transform.resize(cropped_letter, (32,32))
        cropped_letters.append(cropped_letter)

    # load the weights and running extracted text through our network
    import pickle
    import string
    
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('../output/q3_weights.pickle','rb'))
    
    #Running Inference
    for cropped_letter in cropped_letters: 

        h1 = forward(cropped_letter.T.flatten(), params, 'layer1')
        probs = forward(h1.reshape(1, -1), params, 'output', softmax)
        index = np.argmax(probs, axis=1)
        prediction = letters[index]
        plt.imshow(cropped_letter, cmap="Greys")
        plt.show()
        print(prediction[0], end=' ')
        