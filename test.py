import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from roadnet import RoadNet
from data_loader import train_images, labels_segments

NUM_TRAIN = 100
MODEL_CHECKPOINT = 'checkpoints/model_1.weights.hdf5'

counter = 0
num_test = 3

test_images = []
test_ground_truth = []
test_image_idx = []

model = RoadNet().get_model()
model.load_weights(MODEL_CHECKPOINT)

fig, ax = plt.subplots(3,3, figsize=(3,3))

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
    map_ = np.argmax(map_, axis=2)
    
    gt = np.argmax(gt, axis=2)

    ax[idx][0].imshow(img)
    ax[idx][1].imshow(gt)
    ax[idx][2].imshow(map_)

plt.show()
