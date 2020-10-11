import cv2
import json
import rasterio

from rasterio import features
from rasterio import Affine
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--image', required=True, help='Path to rasterized image')
parser.add_argument('--top', required=True, help='Top coordinate of the given image')
parser.add_argument('--left', required=True, help='Left coordinate of the given image')
parser.add_argument('--bottom', required=True, help='Bottom coordinate of the given image')
parser.add_argument('--right', required=True, help='Right coordinate of the given image')
args = vars(parser.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = image != 255
H, W = image.shape

top = float(args['top'])
right = float(args['right'])
bottom = float(args['bottom'])
left = float(args['left'])

X_ratio = (right - left)/float(W)
Y_ratio = (top - bottom)/float(H)

transform = Affine(X_ratio, 0, left, 0, Y_ratio, top)

### Extract shapes from the positive source image ###
shapes = features.shapes(image, mask=mask, transform=transform)
shapes = list(shapes)

### Convert the shapes into geojson ###
results = []
for (g, v) in shapes:
    shape = {
        'type' : 'Feature',
        'properties' : {'raster_val' : v},
        'geometry' : g
    }

    results.append(shape)

collection = {
    'type' : 'FeatureCollection',
    'features' : results
}

print('[INFO] Parsing the image into geojson ... ')
with open('geojson/object.json', 'w') as dst:
    json.dump(collection, dst, indent=4)
