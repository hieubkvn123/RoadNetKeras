import time
import json

import rasterio
import rasterio.features
import rasterio.warp
import pprint

from osgeo import gdal, osr

src_filename = '../data/2/segmentation.png'
dst_filename = 'destination.tif'

feature_collection = {}
feature_collection['type'] = 'FeatureCollection'
feature_collection['features'] = []

def to_geojson(file_name):
    with rasterio.open(file_name) as src:
        blue = src.read(3)

    mask = blue != 255
    shapes = rasterio.features.shapes(blue, mask=mask, transform=src.transform)

    object_ = {}
    object_['type'] = 'Feature'
    
    counter = 0
    for shape in shapes:
        print('[*] Sparsing shape #{:04d}'.format(counter + 1))
        object_['type'] = 'Feature'
        object_['geometry'] = shape
        object_['properties'] = {'prop{:01d}'.format(counter) : 'prop{:01d}'.format(counter)}
        
        feature_collection['features'].append(object_)
        counter += 1

    feature_collection['properties'] = {'props':'val'}
    
    filename = 'geojson/obj_%.2f.geojson' % time.time()
    print('[*] Writing to file : %s' % filename)
    json.dump(feature_collection, open(filename, 'w'), indent=2)


to_geojson(src_filename)
