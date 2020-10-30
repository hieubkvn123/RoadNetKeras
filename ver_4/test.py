import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from roadnet import RoadNet
from data_loader import train_images, labels_segments
from tensorflow.keras.models import Model

NUM_TRAIN = 100
MODEL_CHECKPOINT = 'checkpoints/model_4.weights.hdf5'

counter = 0
num_test = 3

test_images = []
test_ground_truth = []
test_image_idx = []

model = RoadNet().get_model()
model.load_weights(MODEL_CHECKPOINT)
model = Model(model.inputs, [
        model.get_layer('surface_final_output').output,\
        model.get_layer('line_final_output').output,\
        model.get_layer('edge_final_output').output\
])

fig, ax = plt.subplots(3,4, figsize=(30,30))

while(counter < num_test):
    test_id = np.random.randint(0, NUM_TRAIN)
    if(test_id in test_image_idx):
        continue

    test_image_idx.append(test_id)

    test_img = train_images[test_id]
    test_gt = labels_segments[test_id]
    
    test_images.append(test_img)
    test_ground_truth.append(test_gt)

    counter += 1

for idx, (img, gt) in enumerate(zip(test_images, test_ground_truth)):
    map_ = model.predict(np.array([img]))[0][0]
    line = model.predict(np.array([img]))[1][0]
    edge = model.predict(np.array([img]))[2][0]
    
    map_[map_ > 0.5] = 1
    map_[map_ < 0.5] = 0

    line[line > 0.5] = 1
    line[line < 0.5] = 0
    
    edge[edge > 0.5] = 1
    edge[edge < 0.5] = 0
    
    ax[idx][0].imshow(gt)
    ax[idx][1].imshow(map_)
    ax[idx][2].imshow(line)
    ax[idx][3].imshow(edge)

ax[0][0].set_title("Segmentation ground truth")
ax[0][1].set_title("Segmentation prediction")
ax[0][2].set_title("Centerline prediction")
ax[0][3].set_title("Edge prediction")

plt.show()
