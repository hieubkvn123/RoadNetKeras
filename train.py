import os
import cv2
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from roadnet import RoadNet

DATA_DIR = 'data/'
TRAIN_IMG_PICKLE = 'data/img.pickle'
TRAIN_SEG_PICKLE = 'data/segments.pickle'
TRAIN_EDG_PICKLE = 'data/edges.pickle'
TRAIN_CEN_PICKLE = 'data/centerlines.pickle'

TRAIN_SET = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
TEST_SET  = [1,16,17,18,19,20]

train_images = []
test_images  = []

labels_segments    = []
labels_edges       = []
labels_centerlines = []

def cropping_images(img, crop_size=(128,128)):
    crops = []

    H, W, C = img.shape

    ### Get the ratio to resize ###
    ratio_h = int(H/128)
    ratio_w = int(W/128)

    ### get the refined resize dimensions ###
    resized_dimensions = (128 * ratio_w , 128 * ratio_h)

    ### resize the image ###
    img_resize = cv2.resize(img, resized_dimensions)

    ### Divide the images into chunks of 128 x 128 squares ###
    for i in range(ratio_h):
        for j in range(ratio_w):
            crop = img_resize[i*128: (i+1)*128, j*128:(j+1)*128]

            crops.append(crop)

    return crops

if(not os.path.exists(TRAIN_IMG_PICKLE) or 
        not os.path.exists(TRAIN_SEG_PICKLE) or
        not os.path.exists(TRAIN_CEN_PICKLE)):
    for entry in TRAIN_SET:
        abs_img_path = DATA_DIR + ("/%d/" % entry) + ("Ottawa-%d.tif" % entry)
        abs_surface_path = DATA_DIR + ("/%d/" % entry) + "segmentation.png"
        abs_edge_path = DATA_DIR + ("/%d/" % entry) + "edge.png"
        abs_centerline_path = DATA_DIR + ("/%d/" % entry) + "centerline.png"

        print("[INFO] Processing training image with id %d ..." % entry)

        print(abs_img_path)
        img = cv2.imread(abs_img_path)
        edge = cv2.imread(abs_edge_path)
        surface = cv2.imread(abs_surface_path)
        centerline = cv2.imread(abs_centerline_path)

        img_crops = cropping_images(img)
        surface_crops = cropping_images(surface) 
        edge_crops = cropping_images(edge)
        centerline_crops = cropping_images(centerline)

        train_images.extend(img_crops)
        labels_segments.extend(surface_crops)
        labels_edges.extend(edge_crops)
        labels_centerlines.extend(centerline_crops)

    ### Serializing the data ###
    print('[INFO] Serializing data ...')
    pickle.dump(train_images, open(TRAIN_IMG_PICKLE, 'wb'))
    pickle.dump(labels_segments, open(TRAIN_SEG_PICKLE, 'wb'))
    pickle.dump(labels_edges, open(TRAIN_EDG_PICKLE, 'wb'))
    pickle.dump(labels_centerlines, open(TRAIN_CEN_PICKLE, 'wb'))
else:
    print('[INFO] Loading data ...')
    train_images = pickle.load(open(TRAIN_IMG_PICKLE, 'rb'))
    labels_segments = pickle.load(open(TRAIN_SEG_PICKLE, 'rb'))
    labels_edges = pickle.load(open(TRAIN_EDG_PICKLE, 'rb'))
    labels_centerlines = pickle.load(open(TRAIN_CEN_PICKLE, 'rb'))

train_images = np.array(train_images)
labels_segments = np.array(labels_segments)
labels_edges = np.array(labels_edges)
labels_centerlines = np.array(labels_centerlines)
print('=============================================================')
print('[INFO] Summary : %d cropped images,\n %d cropped surfaces,\n %d cropped edges,\n %d cropped centerlines' \
        % (train_images.shape[0], labels_segments.shape[0], labels_edges.shape[0], labels_centerlines.shape[0])) 
 
random_id = np.random.randint(0, train_images.shape[0])
cv2.imshow("Sample Input", train_images[random_id])
cv2.imshow("Sample Edge", labels_edges[random_id])
cv2.imshow("Sample Surface", labels_segments[random_id])
cv2.imshow("Sample Centerline", labels_centerlines[random_id])
cv2.waitKey(0)

net = RoadNet()
model = net.get_model()
