import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd

### Loading data ###
from data_loader import train_images, labels_segments, labels_edges, labels_centerlines
from data_loader import test_images, test_labels_segments, test_labels_edges, test_labels_centerlines

from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger
from argparse import ArgumentParser
from roadnet import RoadNet

parser = ArgumentParser()
parser.add_argument('-d', '--dir', required=False, help='Path to data folder')
parser.add_argument('-n', '--n_train', required=False, help='Number of training partions you want to train')
parser.add_argument('-i', '--iterations', required=False, help='Number of epochs to train')
parser.add_argument('-p', '--patience', required=False, help='Number of patient epochs without improvement on loss')
parser.add_argument('-c', '--checkpoint', required=False, help='Path to checkpoint file')
args = vars(parser.parse_args())

DATA_DIR = 'data/'
NUM_TRAIN_IMG=100
EPOCHS = 1000
BATCH_SIZE=32
PATIENCE=15
MODEL_CHECKPOINT = 'checkpoints/model_1.weights.hdf5'

### Create checkpoints directory if not exists ###
if(not os.path.exists('checkpoints')):
    os.mkdir('checkpoints')

if(args['dir']): DATA_DIR=args['dir']
if(args['n_train']): NUM_TRAIN_IMG=args['n_train']
if(not args['n_train']): NUM_TRAIN_IMG=train_images.shape[0]
if(args['iterations']): EPOCHS=args['iterations']
if(args['patience']): PATIENCE=args['patience']
if(args['checkpoint']): MODEL_CHECKPOINT=args['checkpoint']

print('=============================================================')
print('[INFO] Summary : %d cropped images,\n %d cropped surfaces,\n %d cropped edges,\n %d cropped centerlines' \
        % (train_images.shape[0], labels_segments.shape[0], labels_edges.shape[0], labels_centerlines.shape[0])) 
 
net = RoadNet()
model = net.get_model()
print(model.summary())
if(os.path.exists(MODEL_CHECKPOINT)):
    model.load_weights(MODEL_CHECKPOINT)
    print('[INFO] Transfer learning from check point. ..')

def lr_decay(i, lr):
    if( i >= 160):
        return 1e-5
    elif( i >= 120):
        return 5e-5
    elif( i >= 80 ):
        return 1e-4
    elif( i >= 40):
        return 5e-4
    elif( i >= 10):
        return 1e-3
    else:
        return lr 

callbacks = [
    ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True),
    EarlyStopping(patience=PATIENCE, verbose=1),
    CSVLogger('training.log.csv', append=True),
    LearningRateScheduler(lr_decay)
]

balanced_loss = net.weighted_binary_crossentropy()
balanced_loss_with_l2 = net.weighted_binary_crossentropy_with_l2()
weighted_ce_with_logits = net.cross_entropy_balanced

losses = {
    'surface_final_output' : balanced_loss_with_l2,# net.weighted_binary_crossentropy,
    'edge_final_output' : balanced_loss_with_l2,# net.weighted_binary_crossentropy,
    'line_final_output' : balanced_loss_with_l2,# net.weighted_binary_crossentropy
    'surface_side_output_1' : balanced_loss, 
    'surface_side_output_2' : balanced_loss,
    'surface_side_output_3' : balanced_loss,
    'surface_side_output_4' : balanced_loss,
    'surface_side_output_5' : balanced_loss,

    'edge_side_output_1' : balanced_loss, 
    'edge_side_output_2' : balanced_loss,
    'edge_side_output_3' : balanced_loss,
    'edge_side_output_4' : balanced_loss,

    'line_side_output_1' : balanced_loss, 
    'line_side_output_2' : balanced_loss,
    'line_side_output_3' : balanced_loss,
    'line_side_output_4' : balanced_loss
}

'''
losses = {
    'surface_final_output' : 'binary_crossentropy',# net.weighted_binary_crossentropy,
    'edge_final_output' : 'binary_crossentropy',# net.weighted_binary_crossentropy,
    'line_final_output' : 'binary_crossentropy',# net.weighted_binary_crossentropy
    'surface_side_output_1' : 'binary_crossentropy', 
    'surface_side_output_2' : 'binary_crossentropy',
    'surface_side_output_3' : 'binary_crossentropy',
    'surface_side_output_4' : 'binary_crossentropy',
    'surface_side_output_5' : 'binary_crossentropy',

    'edge_side_output_1' : 'binary_crossentropy', 
    'edge_side_output_2' : 'binary_crossentropy',
    'edge_side_output_3' : 'binary_crossentropy',
    'edge_side_output_4' : 'binary_crossentropy',

    'line_side_output_1' : 'binary_crossentropy', 
    'line_side_output_2' : 'binary_crossentropy',
    'line_side_output_3' : 'binary_crossentropy',
    'line_side_output_4' : 'binary_crossentropy'
}
'''

loss_weights = {
    'surface_final_output' : 1,
    'edge_final_output' : 1,
    'line_final_output' : 1,

    'surface_side_output_1' : 0.1,
    'surface_side_output_2' : 0.2,
    'surface_side_output_3' : 0.3,
    'surface_side_output_4' : 0.4,
    'surface_side_output_5' : 0.5,
 
    'edge_side_output_1' : 0.1,
    'edge_side_output_2' : 0.2,
    'edge_side_output_3' : 0.3,
    'edge_side_output_4' : 0.4,

    'line_side_output_1' : 0.1,
    'line_side_output_2' : 0.2,
    'line_side_output_3' : 0.3,
    'line_side_output_4' : 0.4 
}

y = {
     'surface_final_output' : labels_segments,
     'edge_final_output' : labels_edges,
     'line_final_output' : labels_centerlines,

     'surface_side_output_1' : labels_segments,
     'surface_side_output_2' : labels_segments,
     'surface_side_output_3' : labels_segments,
     'surface_side_output_4' : labels_segments,
     'surface_side_output_5' : labels_segments,

     'edge_side_output_1' : labels_edges, 
     'edge_side_output_2' : labels_edges, 
     'edge_side_output_3' : labels_edges, 
     'edge_side_output_4' : labels_edges,

     'line_side_output_1' : labels_centerlines,
     'line_side_output_2' : labels_centerlines,
     'line_side_output_3' : labels_centerlines,
     'line_side_output_4' : labels_centerlines 
}

y_test = {
     'surface_final_output' : test_labels_segments,
     'edge_final_output' : test_labels_edges,
     'line_final_output' : test_labels_centerlines,

     'surface_side_output_1' : test_labels_segments,
     'surface_side_output_2' : test_labels_segments,
     'surface_side_output_3' : test_labels_segments,
     'surface_side_output_4' : test_labels_segments,
     'surface_side_output_5' : test_labels_segments,

     'edge_side_output_1' : test_labels_edges, 
     'edge_side_output_2' : test_labels_edges, 
     'edge_side_output_3' : test_labels_edges, 
     'edge_side_output_4' : test_labels_edges,

     'line_side_output_1' : test_labels_centerlines,
     'line_side_output_2' : test_labels_centerlines,
     'line_side_output_3' : test_labels_centerlines,
     'line_side_output_4' : test_labels_centerlines 
}

adam = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.9)
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
mean_iou = MyMeanIOU(num_classes=2)
model.compile(optimizer=adam, loss=losses, loss_weights=loss_weights, metrics=[mean_iou])
history = model.fit(train_images, y=y, validation_split=0.3, epochs=EPOCHS, callbacks=callbacks, batch_size=BATCH_SIZE)

