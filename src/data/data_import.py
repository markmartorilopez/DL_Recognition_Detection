import numpy as np
np.random.seed(2021)
import os
import glob2 as glob
import cv2

import warnings
warnings.filterwarnings("ignore")



# PARAMETERS
#----------
# img_path : path
#            path of the image to be resized

def resize_image(img_path):
    # read image file
    img = cv2.imread(img_path)

    # Resize image to be 32 by 32
    img_resized = cv2.resize(img, (32,32), cv2.INTER_LINEAR)

    return img_resized


def load_training_samples():
    '''
    Loading training samples and resize them
    '''
    # Variables to hold the training input and output variables
    train_input_variables = []
    train_input_variables_id = []
    train_label = []
    # Scanning all images in each folder of a fish type
    print('Start Reading Train Images')
    folders = ['butterfly', 'cat', 'dog', 'horse', 'squirrel', 'spider', 'sheep', 'cow','chicken','elephant']
    for fld in folders:
        folder_index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, folder_index))
        imgs_path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(imgs_path)
        for file in files:
            file_base = os.path.basename(file)
            # Resize the image
            resized_img = resize_image(file)
            # Appending the processed image to the input/output variables of the classifier
            train_input_variables.append(resized_img)
            train_input_variables_id.append(file_base)
            train_label.append(folder_index)
    return train_input_variables, train_input_variables_id, train_label

def load_testing_samples():
    # Scanning images from the test folder
    imgs_path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(imgs_path))
    # Variables to hold the testing samples
    testing_samples = []
    testing_samples_id = []
    #Processing the images and appending them to the array that we have
    for file in files:
       file_base = os.path.basename(file)
       # Image resizing
       resized_img = rezize_image(file)
       testing_samples.append(resized_img)
       testing_samples_id.append(file_base)
    return testing_samples, testing_samples_id

