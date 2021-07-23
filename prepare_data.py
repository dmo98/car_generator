import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tqdm
from tqdm import tqdm
import random
from scipy.io import loadmat

print('Completed importing libraries' + '\n')

IMAGES_DIR = 'car_ims/car_ims'
ANNOTATIONS_FILENAME = 'car_ims/cars_annos.mat'
IMG_SIZE = 64
USE_IMAGES_CROPPED_WITH_BOUNDING_BOXES = True


def get_annotations(annotations_filename):
    """
    Given the annotations file, this function returns a list of the file names, and a list of the bounding boxes
    """
    annotations_data = loadmat(annotations_filename)
    filenames = []
    bb = []
    for annotation in annotations_data['annotations'][0]:
        filename = str(annotation[0]).split('/')[-1][:-2]
        x1 = int(annotation[1])
        y1 = int(annotation[2])
        x2 = int(annotation[3])
        y2 = int(annotation[4])

        filenames.append(filename)
        bb.append((x1, y1, x2, y2))
    return filenames, bb


def create_data(path, filenames, bb, img_height, img_width, crop_with_bounding_boxes=True):
    """
    This function reads every image, crops them using the bounding boxes (to get images containing cars only, without the
    background), resizes them to ('img_height', 'img_width') and returns a list containing these cropped, resized images
    """
    index = 0
    data = []
    for img_file in tqdm(os.listdir(path)):
        if filenames[index] == img_file:
            box = bb[index]
            img_array = cv2.imread(os.path.join(path, img_file))
            if crop_with_bounding_boxes:
                img_array = img_array[box[1]:box[3], box[0]:box[2], :]
            img_array = cv2.resize(img_array, (img_height, img_width))
            data.append(img_array)
        index += 1

    return data


# Call the above functions to create the dataset
filenames, bounding_boxes = get_annotations(ANNOTATIONS_FILENAME)
data = create_data(IMAGES_DIR, filenames, bounding_boxes, IMG_SIZE, IMG_SIZE, crop_with_bounding_boxes=USE_IMAGES_CROPPED_WITH_BOUNDING_BOXES)
print('There are {} images in the directory containing the data'.format(len(data)) + '\n')
# Store the data in a numpy array
X = np.array(data).reshape((-1, IMG_SIZE, IMG_SIZE, 3))
# Save the data in a .npy file
if USE_IMAGES_CROPPED_WITH_BOUNDING_BOXES:
    np.save('data-({},{},3).npy'.format(IMG_SIZE, IMG_SIZE), X)
else:
    np.save('uncropped-data-({},{},3).npy'.format(IMG_SIZE, IMG_SIZE), X)
print('Saved the data in a .npy file')

print('Displaying 10 random images: ')
random.shuffle(data)
for i in range(10):
    plt.imshow(data[i][:, :, ::-1])
    plt.show()
