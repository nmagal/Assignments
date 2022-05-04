from os.path import join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import util
import visual_words
import visual_recog
from opts import get_opts
import pdb


def main():
    opts = get_opts()

    # Q1.1 - Extracting Filter Responses 
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    # Q1.2 - Creating Visual Words 
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # Q1.3 - Computing Visual Words 
    img_path = join(opts.data_dir, 'desert/sun_aaqyzvrweabdxjzo.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(wordmap)
    
    # Q2.1-2.4 - Building Recognition System 
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5 - Quantitative Evaluation
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    #Printing and saving our results
    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

if __name__ == '__main__':
    main()
