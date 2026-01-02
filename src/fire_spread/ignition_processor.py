import numpy as np
from numpy.typing import NDArray
import pandas as pd
import geopandas as gpd
import json
from scipy.interpolate import griddata
import rasterio
from rasterio import features
from affine import Affine
import matplotlib.pyplot as plt

# Support both package import (fire_spread.*) and direct script execution.
try:
    from .terrain import Terrain
    from .coordinates_transform import CoordinatesTransform
except ImportError:  # pragma: no cover
    from terrain import Terrain
    from coordinates_transform import CoordinatesTransform


class IgnitionProcessor:
    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.df = None
        self.mask = None
        self.ignition_grid = None
        self.tr = CoordinatesTransform(terrain.crs, terrain.get_transform())

    def load_and_prepare_data(self, json_path, bbox=None, timedelta=pd.Timedelta(minutes=1)):
        """
        :param self: Description
        :param json_path: Description
        :param bbox: x_min, y_min, x_max, y_max
        :param timedelta: pd.Timedelta - smallest interval between two ignition events
        :return: Description
        :rtype: DataFrame | Series[Any]
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(
            df['acq_date'] + ' ' + df['acq_time'], 
            format='%Y-%m-%d %H%M'
        )

        rows, cols = self.tr.index(df['longitude'].tolist(), df['latitude'].tolist())
        df['X'], df['Y'] = cols, rows

        self.df = self._filter_by_bbox(df, bbox)

        if self.df.empty:
            raise ValueError("No fire points found within the specified bounding box.")

        time_zero = self.df['timestamp'].min()
        print(time_zero)
        self.df['ignition_min'] = ((self.df['timestamp'] - time_zero) / timedelta).astype(int)
        
        return self.df

    def _filter_by_bbox(self, df, bbox):
        if bbox is None:
            return df.copy()

        x_min, y_min, x_max, y_max = bbox
        mask = pd.Series(True, index=df.index)

        if x_min is not None:
            mask &= (df['X'] >= x_min)
        if x_max is not None:
            mask &= (df['X'] <= x_max)
        if y_min is not None:
            mask &= (df['Y'] >= y_min)
        if y_max is not None:
            mask &= (df['Y'] <= y_max)
        
        return df[mask].copy()

    def create_boundary_mask(self, gdf_path):
        if self.tr is None:
            raise AttributeError('Transform must be valid (not null)')
        gdf = gpd.read_file(gdf_path)

        h, w = self.terrain.get_dimensions()

        self.mask = self.tr.map_polygons_to_grid(
            gdf,
            width=w,
            height=h
        )
        return self.mask

    def interpolate_ignition_time(self, method='linear') -> NDArray:
        """
        :param method: Interpolation method for scipy.interpolate.griddata
        :return: 2D Array of ignition time from previously set time
        :rtype: NDArray
        """
        if self.df is None or self.mask is None:
            raise ValueError("Data must be loaded and mask must be created first")

        points = self.df[['Y', 'X']].to_numpy()
        values = self.df['ignition_min'].to_numpy()
        h, w = self.terrain.get_dimensions()

        grid_y, grid_x = np.mgrid[0:h:complex(0, h), 0:w:complex(0, w)]
        
        grid_z = griddata(points, values, (grid_y, grid_x), method=method)
        
        self.ignition_grid = np.where(self.mask == 1, grid_z, np.nan)
        return self.ignition_grid

    def plot_results(self, show_points=True):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.terrain.biomass, cmap='gray', alpha=0.4)
        
        im = plt.imshow(self.ignition_grid, cmap='plasma_r', alpha=0.8)
        plt.colorbar(im, label='Time since fire start')
        
        if show_points:
            plt.scatter(self.df['X'], self.df['Y'], c='red', s=10, label='Detected Points')

        plt.title("Reconstructed Ignition Time Map")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import os
    import pathlib

    ROOT_PATH = pathlib.Path(os.path.realpath(__file__)).parents[2]
    TIF_PATH = ROOT_PATH / 'data' / 'las_20m_resolution.tif'
    terrain = Terrain(str(TIF_PATH))
    
    ip = IgnitionProcessor(terrain)

    # FIRE_DATA =  ROOT_PATH / 'data' / 'fire_archive_SV-C2_675228.json'
    FIRE_DATA =  ROOT_PATH / 'data' / 'fire_archive_M-C61_675224.json'
    ip.load_and_prepare_data(str(FIRE_DATA), bbox=(0, 0, None, 600), timedelta=pd.Timedelta(hours=1))

    MASK_PATH = ROOT_PATH / 'data' / '1a6cd4865f484fb48f8ba4ea97a6e0d1.json'
    ip.create_boundary_mask(str(MASK_PATH))
    grid = ip.interpolate_ignition_time()
    print(f'The grid shape is: {grid.shape}')
    plt.imshow(grid)
    plt.show()
    # ip.plot_results(show_points=False)

    



