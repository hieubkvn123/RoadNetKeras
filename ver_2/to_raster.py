import time
import json

import rasterio
import rasterio.features
import rasterio.warp
import pprint

from osgeo import gdal, osr

src_filename = '../data/2/segmentation.png'
dst_filename = 'destination.tif'

src_ds = gdal.Open(src_filename)
format = "GTiff"
driver = gdal.GetDriverByName(format)

# Open destination dataset
dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)

# Specify raster location through geotransform array
# (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
# Scale = size of one pixel in units of raster projection
# this example below assumes 100x100
gt = [-75.777511597, 7936, 0, 45.340576172, 0, 4864]

dst_ds.SetGeoTransform(gt)

epsg = 4326
srs = osr.SpatialReference()
srs.ImportFromEPSG(epsg)
dest_wkt = srs.ExportToWkt()

dst_ds.SetProjection(dest_wkt)

dst_ds = None
src_ds = None

feature_collection = {}
feature_collection['type'] = 'FeatureCollection'
feature_collection['properties'] = {'props':'val'}
feature_collection['features'] = []

def to_geojson(file_name):
    with rasterio.open(file_name) as src:
        blue = src.read(3)

    mask = blue != 255
    shapes = rasterio.features.shapes(blue, mask=mask)

    object_ = {}
    object_['type'] = 'Feature'
    
    counter = 0
    for shape in shapes:
        print('[*] Sparsing shape #{:04d}'.format(counter + 1))
        object_['type'] = 'Feature'
        object_['properties'] = {'prop{:01d}'.format(counter) : 'prop{:01d}'.format(counter)}
        object_['geometry'] = shape

        feature_collection['features'].append(object_)
        counter += 1

    filename = 'geojson/obj_%.2f.geojson' % time.time()
    print('[*] Writing to file : %s' % filename)
    json.dump(feature_collection, open(filename, 'w'), indent=2)


to_geojson(dst_filename)
