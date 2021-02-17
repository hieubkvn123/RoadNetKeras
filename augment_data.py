import os
import cv2
import numpy as np

from skimage import exposure
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data-dir', help='Path to dataset', required=True)
parser.add_argument('--ref-img-path', help='Path to reference image', required=True)
args = vars(parser.parse_args())

'''
This script augment the original Ottawa data using histogram matching
the original GG Earth dataset with a reference image from Bing Map
'''

ref_image = cv2.imread(args['ref_img_path'])
data_dir  = args['data_dir']

def _hist_matching(image, ref_image):
    assert isinstance(image, np.ndarray)
    assert isinstance(ref_image, np.ndarray)

    new_image = exposure.match_histograms(image, ref_image, multichannel=True)
    return new_image

print('[INFO] Starting data augmentation ... ')
for (dir_, dirs, files) in os.walk(data_dir):
    if(dir_ != data_dir):
        for file_ in files:
            if(not file_.endswith('.tif')):
                continue

            abs_path = os.path.join(dir_, file_)
            image = cv2.imread(abs_path)
            new_image = _hist_matching(image, ref_image)

            file_name = file_.split('.')[0]
            new_file_name = file_name + '_bingmap.png'
            new_abs_path = os.path.join(dir_, new_file_name)

            cv2.imwrite(new_abs_path, new_image)
            print('[INFO] Wrote image to %s' % new_abs_path)
