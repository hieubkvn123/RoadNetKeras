import os
import cv2
import pickle
import numpy as np
import tensorflow as tf

NUM_TRAIN_IMG = 1000 
DATA_DIR = '../data/'
TRAIN_IMG_PICKLE = 'data/img.pickle'
TRAIN_SEG_PICKLE = 'data/segments.pickle'
TRAIN_EDG_PICKLE = 'data/edges.pickle'
TRAIN_CEN_PICKLE = 'data/centerlines.pickle'

TEST_IMG_PICKLE = 'data/test_img.pickle'
TEST_SEG_PICKLE = 'data/test_segments.pickle'
TEST_EDG_PICKLE = 'data/test_edges.pickle'
TEST_CEN_PICKLE = 'data/test_centerlines.pickle'

TRAIN_SET = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
TEST_SET  = [1,16,17,18,19,20]
H, W, C, C_ = 512, 512, 3, 1

train_images = []
test_images  = []

labels_segments    = []
labels_edges       = []
labels_centerlines = []

test_labels_segments = []
test_labels_edges = []
test_labels_centerlines = []

def cropping_images(img, crop_size=(512,512)):
    crops = []

    H, W = img.shape[0], img.shape[1]

    ### Get the ratio to resize ###
    ratio_h = int(H/crop_size[0])
    ratio_w = int(W/crop_size[1])

    ### get the refined resize dimensions ###
    resized_dimensions = (crop_size[1] * ratio_w , crop_size[0] * ratio_h)

    ### resize the image ###
    img_resize = cv2.resize(img, resized_dimensions)

    ### Divide the images into chunks of 128 x 128 squares ###
    for i in range(ratio_h):
        for j in range(ratio_w):
            crop = img_resize[i*crop_size[0]: (i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]]

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
        edge = cv2.cvtColor(cv2.imread(abs_edge_path), cv2.COLOR_BGR2GRAY)
        surface = cv2.cvtColor(cv2.imread(abs_surface_path), cv2.COLOR_BGR2GRAY)
        centerline = cv2.cvtColor(cv2.imread(abs_centerline_path), cv2.COLOR_BGR2GRAY)

        edge[edge < 250] = 1
        edge[edge >= 250] = 0

        surface[surface < 250] = 1
        surface[surface >= 250] = 0

        centerline[centerline < 250] = 1
        centerline[centerline >= 250] = 0

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

if(NUM_TRAIN_IMG < 0): NUM_TRAIN_IMG=len(train_images)
print(np.array(train_images).shape)
train_images = np.array(train_images)[:NUM_TRAIN_IMG].reshape(-1, H, W, C)
labels_segments = np.array(labels_segments)[:NUM_TRAIN_IMG]
labels_edges = np.array(labels_edges)[:NUM_TRAIN_IMG]
labels_centerlines = np.array(labels_centerlines)[:NUM_TRAIN_IMG]

labels_segments = tf.one_hot(labels_segments, depth=2)
labels_edges = tf.one_hot(labels_edges, depth=2)
labels_centerlines = tf.one_hot(labels_centerlines, depth=2)

if(not os.path.exists(TEST_IMG_PICKLE) or 
        not os.path.exists(TEST_SEG_PICKLE) or
        not os.path.exists(TEST_CEN_PICKLE)):
    for entry in TEST_SET:
        abs_img_path = DATA_DIR + ("/%d/" % entry) + ("Ottawa-%d.tif" % entry)
        abs_surface_path = DATA_DIR + ("/%d/" % entry) + "segmentation.png"
        abs_edge_path = DATA_DIR + ("/%d/" % entry) + "edge.png"
        abs_centerline_path = DATA_DIR + ("/%d/" % entry) + "centerline.png"

        print("[INFO] Processing testing image with id %d ..." % entry)

        print(abs_img_path)
        img = cv2.imread(abs_img_path)
        edge = cv2.cvtColor(cv2.imread(abs_edge_path), cv2.COLOR_BGR2GRAY)
        surface = cv2.cvtColor(cv2.imread(abs_surface_path), cv2.COLOR_BGR2GRAY)
        centerline = cv2.cvtColor(cv2.imread(abs_centerline_path), cv2.COLOR_BGR2GRAY)

        edge[edge < 250] = 1
        edge[edge >= 250] = 0

        surface[surface < 250] = 1
        surface[surface >= 250] = 0

        centerline[centerline < 250] = 1
        centerline[centerline >= 250] = 0

        img_crops = cropping_images(img)
        surface_crops = cropping_images(surface) 
        edge_crops = cropping_images(edge)
        centerline_crops = cropping_images(centerline)

        test_images.extend(img_crops)
        test_labels_segments.extend(surface_crops)
        test_labels_edges.extend(edge_crops)
        test_labels_centerlines.extend(centerline_crops)

    ### Serializing the data ###
    print('[INFO] Serializing data ...')
    pickle.dump(test_images, open(TEST_IMG_PICKLE, 'wb'))
    pickle.dump(test_labels_segments, open(TEST_SEG_PICKLE, 'wb'))
    pickle.dump(test_labels_edges, open(TEST_EDG_PICKLE, 'wb'))
    pickle.dump(test_labels_centerlines, open(TEST_CEN_PICKLE, 'wb'))
else:
    print('[INFO] Loading data ...')
    test_images = pickle.load(open(TEST_IMG_PICKLE, 'rb'))
    test_labels_segments = pickle.load(open(TEST_SEG_PICKLE, 'rb'))
    test_labels_edges = pickle.load(open(TEST_EDG_PICKLE, 'rb'))
    test_labels_centerlines = pickle.load(open(TEST_CEN_PICKLE, 'rb'))

test_images = np.array(test_images).reshape(-1, H, W, C)
test_labels_segments = np.array(test_labels_segments)
test_labels_edges = np.array(test_labels_edges)
test_labels_centerlines = np.array(test_labels_centerlines)

test_labels_segments = tf.one_hot(test_labels_segments, depth=2)
test_labels_edges = tf.one_hot(test_labels_edges, depth=2)
test_labels_centerlines = tf.one_hot(test_labels_centerlines, depth=2)
