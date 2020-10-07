import rasterio
import rasterio.features
import rasterio.warp

def to_geojson(image_file):  
    profile = {'driver': 'GTiff', 'height': 100, 'width': 100, 'count': 1, 'dtype': rasterio.uint8}
    with rasterio.open(image_file, **profile) as dataset:
        mask = dataset.dataset_mask()

        ### Extract feature shapes and values from the array ###
        for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
            print(dataset.crs)
            geom = rasterio.warp.transform_geom(
                dataset.crs,
                'EPSG:3857',
                geom,
                precision=6
            )

            print(geom)
