import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from roadnet import RoadNet
from tensorflow.keras.models import Model
from data_loader import train_images, labels_segments

parser = ArgumentParser()
parser.add_argument('-m', '--model', required=False, help='Path to the model checkpoint')
parser.add_argument('-i', '--image', required=False, help='Path to the testing image')
args = vars(parser.parse_args())

MODEL_CHECKPOINT = 'checkpoints/model_1.weights.hdf5'
TEST_IMG = '../data/1/Ottawa-1.tif'

if(args['model']) : MODEL_CHECKPOINT = args['model']
if(args['image']) : TEST_IMG = args['image']

TEST_LBL = '../data/' + TEST_IMG.split('/')[2] + "/segmentation.png"

original = cv2.imread(TEST_IMG)
gt_segment = cv2.imread(TEST_LBL)

model = RoadNet().get_model()
model.load_weights(MODEL_CHECKPOINT)
model = Model(model.inputs, [model.get_layer('surface_final_output').output, model.get_layer('line_final_output').output])

def crop_image(img, crop_size=(128,128)):
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

    return crops, ratio_h, ratio_w

img = cv2.imread(TEST_IMG)
crops, ratio_h, ratio_w = crop_image(img)

crops = np.array(crops)

print('[INFO] Running full prediction on segmentation')
full_image = None
for i in range(ratio_h):
    horizontal_image = None
    for j in range(ratio_w):
        index = ratio_w * i + j

        map_ = model.predict(np.array([crops[index]]))[0][0]
        map_[map_ < 0.5] = 0
        map_[map_ > 0.5] = 1

        if(j == 0):
            horizontal_image = map_
        else:
            horizontal_image = cv2.hconcat([horizontal_image, map_])

    if(i == 0):
        full_image = horizontal_image
    else:
        full_image = cv2.vconcat([full_image, horizontal_image])

print('[INFO] Running full prediction on centerlines')
full_image_line = None
for i in range(ratio_h):
    horizontal_image = None
    for j in range(ratio_w):
        index = ratio_w * i + j

        map_ = model.predict(np.array([crops[index]]))[1][0]
        map_[map_ < 0.5] = 0
        map_[map_ > 0.5] = 1

        if(j == 0):
            horizontal_image = map_
        else:
            horizontal_image = cv2.hconcat([horizontal_image, map_])

    if(i == 0):
        full_image_line = horizontal_image
    else:
        full_image_line = cv2.vconcat([full_image_line, horizontal_image])

basename = os.path.basename(TEST_IMG)
filename = basename.split('.')[0]
save_file_name = "sample_predictions/full_prediction_" + filename + ".jpg"

full_image *= 255
full_image = full_image.astype(np.uint8)

full_image_line *= 255
full_image_line = full_image_line.astype(np.uint8)

print("Writing Image to " + save_file_name)

cv2.imwrite(save_file_name, full_image)

fig, ax = plt.subplots(1,3, figsize=(45, 15))
ax[0].imshow(gt_segment)
ax[1].imshow(full_image)
ax[2].imshow(full_image_line)
plt.show()
