import numpy as np
from numpy.typing import NDArray
import pandas as pd
import geopandas as gpd
import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from terrain import Terrain
from coordinates_transform import CoordinatesTransform


class IgnitionProcessor:
    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.df = None
        self.mask = None
        self.ignition_grid = None  # Stores Unix timestamps
        self.tr = CoordinatesTransform(terrain.crs, terrain.get_transform())

    def load_and_prepare_data(self, json_path, bbox=None):
        """
        Loads JSON data, converts times to Unix timestamps, maps coordinates to grid,
        and filters by bounding box.
        
        :param json_path: Path to the JSON fire data.
        :param bbox: tuple (x_min, y_min, x_max, y_max)
        :return: Processed DataFrame
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        df['timestamp'] = pd.to_datetime(
            df['acq_date'] + ' ' + df['acq_time'], 
            format='%Y-%m-%d %H%M'
        )
        
        df['timestamp_unix'] = df['timestamp'].astype('int64') // 10**9

        rows, cols = self.tr.index(df['longitude'].tolist(), df['latitude'].tolist())
        df['X'], df['Y'] = cols, rows

        self.df = self._filter_by_bbox(df, bbox)

        if self.df.empty:
            raise ValueError("No fire points found within the specified bounding box.")
        
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
        """
        Creates a boolean mask from a GeoJSON/Shapefile.
        """
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
        Interpolates ignition times based on Unix timestamps.
        
        :param method: Interpolation method for scipy.interpolate.griddata
        :return: 2D Array of Unix timestamps (seconds)
        """
        if self.df is None:
            raise ValueError("Data must be loaded first")

        points = self.df[['Y', 'X']].to_numpy()
        values = self.df['timestamp_unix'].to_numpy()
        h, w = self.terrain.get_dimensions()

        grid_y, grid_x = np.mgrid[0:h:complex(0, h), 0:w:complex(0, w)]
        
        grid_z = griddata(points, values, (grid_y, grid_x), method=method)
        
        if self.mask is not None:
            self.ignition_grid = np.where(self.mask == 1, grid_z, np.nan)
        else:
            self.ignition_grid = grid_z
            
        return self.ignition_grid

    def get_timedelta_grid(self, timedelta_unit=pd.Timedelta(minutes=1), start_time=None) -> NDArray:
        """
        Converts the internal Unix timestamp grid into a relative time grid.
        
        :param timedelta_unit: The unit to express the output grid in (e.g., minutes, hours).
        :param start_time: Optional start time (pd.Timestamp or datetime). 
                           If None, uses the earliest timestamp found in the loaded data.
        :return: 2D Array of relative time units (floats or ints depending on calculation)
        """
        if self.ignition_grid is None:
            raise ValueError("Ignition grid has not been calculated yet. Run interpolate_ignition_time first.")

        if start_time is None:
            t0_unix = self.df['timestamp_unix'].min()
        else:
            t0 = pd.to_datetime(start_time)
            t0_unix = t0.value // 10**9

        delta_seconds = self.ignition_grid - t0_unix
        
        unit_seconds = timedelta_unit.total_seconds()
        
        return delta_seconds / unit_seconds

    def plot_results(self, grid=None, title="Ignition Map", show_points=True):
        """
        Helper to plot a specific grid.
        """
        if grid is None:
            grid = self.ignition_grid

        plt.figure(figsize=(12, 8))
        if hasattr(self.terrain, 'biomass'):
            plt.imshow(self.terrain.biomass, cmap='gray', alpha=0.4)
        
        im = plt.imshow(grid, cmap='plasma_r', alpha=0.8)
        plt.colorbar(im, label='Time')
        
        if show_points and self.df is not None:
            plt.scatter(self.df['X'], self.df['Y'], c='red', s=10, label='Detected Points')

        plt.title(title)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    import os
    import pathlib

    ROOT_PATH = pathlib.Path(os.path.realpath(__file__)).parents[2]
    TIF_PATH = ROOT_PATH / 'data' / 'las_20m_resolution.tif'
    
    if TIF_PATH.exists():
        terrain = Terrain(str(TIF_PATH))
        ip = IgnitionProcessor(terrain)

        FIRE_DATA = ROOT_PATH / 'data' / 'fire_archive_SV-C2_675228.json'
        
        ip.load_and_prepare_data(str(FIRE_DATA), bbox=(0, 0, None, 600))

        MASK_PATH = ROOT_PATH / 'data' / '1a6cd4865f484fb48f8ba4ea97a6e0d1.json'
        if MASK_PATH.exists():
             ip.create_boundary_mask(str(MASK_PATH))
        
        unix_grid = ip.interpolate_ignition_time()
        print(f'Unix Grid Shape: {unix_grid.shape}')
        print(f'Sample Unix Time: {np.nanmin(unix_grid)}')

        rel_grid_minutes = ip.get_timedelta_grid(timedelta_unit=pd.Timedelta(hours=1))
        
        custom_start = pd.Timestamp("2023-01-01 12:00:00") 
        # rel_grid_hours = ip.get_timedelta_grid(
        #     timedelta_unit=pd.Timedelta(hours=1), 
        #     start_time=custom_start
        # )

        ip.plot_results(grid=rel_grid_minutes, title="Time since start (minutes)")