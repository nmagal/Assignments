import numpy as np
import cv2
from helper import loadVid
import pdb
import planarH as h
from matchPics import matchPics
from opts import get_opts
import multiprocessing
from itertools import repeat

#Crops our source frames to fit destination frame
def crop_size(source_frames, destination_image_match):
    
    #cropping image to keep video centered
    centered_source = source_frames[:, :, 145:496, :]
    
    #Chopping off black part
    centered_source = centered_source[:, 44:316, :, :]
    centered_source[:,:,:, 0]+= 1
    
    return(centered_source)

#applying homography and then superimposing warped image
def augment_frame(source, book_template, destination):
    opts = get_opts()
    
    #Getting correspondences 
    matches, locs1, locs2 = matchPics(destination, book_template, opts)
    locs1 = h.sort_locs(locs1)
    locs2 = h.sort_locs(locs2)
    locs1, locs2 = h.get_matches(matches, locs1, locs2)
    
    #Getting homography matrix
    homography_matrix, inlier_coordinates = h.computeH_ransac(locs1, locs2, opts)  
    
    #We must warp our source to be of the same size as computer vision textbook cover
    resize_dim = (book_template.shape[1], book_template.shape[0])
    source = cv2.resize(source, resize_dim)  
    warp_size = destination.shape[0:2]
    
    #Applying homography
    warped_source = cv2.warpPerspective(src = source, M=homography_matrix, dsize = warp_size[::-1])
    
    #Superimposing warped image onto scene
    composite_frame = h.compositeH(warped_source, destination)
    
    return(composite_frame)
    
if __name__ == "__main__":
    
    #Reading in images 
    panda_path = ('../data/ar_source.mov')
    book_case_path = ('../data/book.mov')
    template = cv2.imread('../data/cv_cover.jpg')
    
    #Creating frames from our vides
    source_frames = loadVid(panda_path)
    destination_frames = loadVid(book_case_path)[0:len(source_frames)]
    source_frames = crop_size(source_frames,template)
    
    #Turning frames into a list of frames
    source_frames = list(np.moveaxis(source_frames, 0, 0))
    destination_frames = list(np.moveaxis(destination_frames, 0, 0))
     
    #Apply AR to all frames
    with multiprocessing.Pool() as pool:
        composite_frames = pool.starmap(augment_frame,zip(source_frames, repeat(template), destination_frames))
    
    #Writing Video 
    frame_size = composite_frames[0].shape[1], composite_frames[0].shape[0]    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('../output/ar.avi', fourcc, 20.0, frame_size, isColor = True)    
    for frame in composite_frames:
        out.write(frame)       
    out.release()