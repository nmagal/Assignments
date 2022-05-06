'''
Hyperparameters wrapped in argparse
This file contains most of tuanable parameters for this homework


You can change the values by changing their default fields or by command-line
arguments. For example, "python q2_1_4.py --sigma 0.15 --ratio 0.7"
'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW2: Homography')

    # Feature detection (requires tuning) decreaseing sigma reduces matches
    parser.add_argument('--sigma', type=float, default=.15,
                        help='threshold for corner detection using FAST feature detector')
    
    #increasing the ratio causes more matches
    parser.add_argument('--ratio', type=float, default=.8,
                        help='ratio for BRIEF feature descriptor')

    # Ransac 
    parser.add_argument('--max_iters', type=int, default=100,
                        help='the number of iterations to run RANSAC for') #default is 500 
    parser.add_argument('--inlier_tol', type=float, default=2.0,
                        help='the tolerance value for considering a point to be an inlier')  #Default is 2.0 

    opts = parser.parse_args()

    return opts
