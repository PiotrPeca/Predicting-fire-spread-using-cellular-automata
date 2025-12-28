import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as reproject
from rasterio.transform import rowcol
from rasterio import features
from affine import Affine
import geopandas as gpd

class CoordinatesTransform:
    def __init__(self, crs: CRS, transform: Affine):
        if crs is None or transform is None:
            raise ValueError("crs or transform cannot be None")
        self.crs = crs
        self.transform = transform

    def index(self, lons: list[float], lats: list[float]) -> tuple[list[int], list[int]]:
        xs, ys = reproject('EPSG:4326', self.crs, lons, lats)

        rows, cols = rowcol(self.transform, xs, ys)

        return [int(r) for r in rows], [int(c) for c in cols]
    
    def map_polygons_to_grid(self, gdf, width, height):
        gdf = gdf.to_crs(self.crs)

        shapes = [(geom, 1) for geom in gdf.geometry]
        mask = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=self.transform,
            fill=0,          
            all_touched=True 
        )

        return mask
