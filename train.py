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
NUM_TRAIN_IMG=100
EPOCHS = 1000
BATCH_SIZE=32
PATIENCE=15
MODEL_CHECKPOINT = 'checkpoints/model.weights.hdf5'

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

    H, W = img.shape

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
        img = cv2.cvtColor(cv2.imread(abs_img_path), cv2.COLOR_BGR2GRAY)
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

train_images = np.array(train_images)[:NUM_TRAIN_IMG]
labels_segments = np.array(labels_segments)[:NUM_TRAIN_IMG].astype(np.float32)
labels_edges = np.array(labels_edges)[:NUM_TRAIN_IMG].astype(np.float32)
labels_centerlines = np.array(labels_centerlines)[:NUM_TRAIN_IMG].astype(np.float32)

print('=============================================================')
print('[INFO] Summary : %d cropped images,\n %d cropped surfaces,\n %d cropped edges,\n %d cropped centerlines' \
        % (train_images.shape[0], labels_segments.shape[0], labels_edges.shape[0], labels_centerlines.shape[0])) 
 
#random_id = np.random.randint(0, train_images.shape[0])
#cv2.imshow("Sample Input", train_images[random_id])
#cv2.imshow("Sample Edge", labels_edges[random_id])
#cv2.imshow("Sample Surface", labels_segments[random_id])
#cv2.imshow("Sample Centerline", labels_centerlines[random_id])
#cv2.waitKey(0)

net = RoadNet()
model = net.get_model()
print(model.summary())

def lr_decay(i, lr):
    if( i < 10):
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callbacks = [
    ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True),
    EarlyStopping(patience=PATIENCE, verbose=1),
    LearningRateScheduler(lr_decay)
]

balanced_loss = net.weighted_binary_crossentropy()

losses = {
    'surface_final_output' : balanced_loss,# net.weighted_binary_crossentropy,
    'edge_final_output' : balanced_loss,# net.weighted_binary_crossentropy,
    'line_final_output' : balanced_loss# net.weighted_binary_crossentropy
}

loss_weights = {
    'surface_final_output' : 1,
    'edge_final_output' : 1,
    'line_final_output' : 1
}

y = {
     'surface_final_output' : labels_segments,
     'edge_final_output' : labels_edges,
     'line_final_output' : labels_centerlines
}

adam = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9,beta_2=0.999,amsgrad=True)
model.compile(optimizer=adam, loss=losses, loss_weights=loss_weights)
model.fit(train_images, y=y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
