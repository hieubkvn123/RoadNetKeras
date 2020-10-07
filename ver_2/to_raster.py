import rasterio
import rasterio.features
import rasterio.warp

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

def to_geojson(file_name):
    with rasterio.open(file_name) as dataset:
        mask = dataset.dataset_mask()

        for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
            geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)

            print(geom)

to_geojson(dst_filename)
