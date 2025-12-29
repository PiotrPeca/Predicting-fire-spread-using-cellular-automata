import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, transform as get_window_transform
from affine import Affine
import numpy as np


class Terrain:
    BIOMASS_THRESHOLDS = {
        "sparse": 50,
        "normal": 150,
    }
    
    VEGKVOT_THRESHOLDS = {
        "shrub": 40,
        "forest": 60,
    }

    def __init__(self, tiff_path, target_size=None, meters_per_pixel=40):
        self.tiff_path = tiff_path
        self.target_size = target_size
        self.meters_per_pixel = meters_per_pixel
        self.transform = None
        self.crs = None
        # Must be aftter self.transform and self.crs
        self.grid_data = self._load_and_process()

    def _load_and_process(self):
        with rasterio.open(self.tiff_path) as dataset:
            src_res = dataset.transform[0]
            target_res = self.meters_per_pixel if self.meters_per_pixel else src_res
            scale = src_res / target_res

            self.crs = dataset.crs

            if self.target_size:
                tgt_h, tgt_w = self.target_size
                src_h, src_w = tgt_h / scale, tgt_w / scale
                
                col_off = (dataset.width - src_w) // 2
                row_off = (dataset.height - src_h) // 2
                
                window = Window(col_off, row_off, src_w, src_h)
                win_transform = get_window_transform(window, dataset.transform)
                self.transform = win_transform * Affine.scale(1/scale, 1/scale)
                out_shape = (tgt_h, tgt_w)
            else:
                window = None
                self.transform = dataset.transform * Affine.scale(1/scale, 1/scale)
                out_shape = (int(dataset.height * scale), int(dataset.width * scale))

            biomass = dataset.read(5, window=window, out_shape=out_shape, resampling=Resampling.bilinear)
            self.biomass = biomass
            vegetation = dataset.read(7, window=window, out_shape=out_shape, resampling=Resampling.bilinear)
            
            return self._generate_grid_matrix(biomass, vegetation)

    def _classify_fuel_type(self, biomass: float, vegkvot: float) -> str:
        if biomass <= 1:
            return "water"
        
        if vegkvot < self.VEGKVOT_THRESHOLDS["shrub"]:
            veg_type = "cultivated"
        elif vegkvot < self.VEGKVOT_THRESHOLDS["forest"]:
            veg_type = "shrub"
        else:
            veg_type = "forest"
        
        if biomass < self.BIOMASS_THRESHOLDS["sparse"]:
            density = "sparse"
        elif biomass < self.BIOMASS_THRESHOLDS["normal"]:
            density = "normal"
        else:
            density = "dense"
        
        return f"{veg_type}_{density}"

    def _generate_grid_matrix(self, biomass_layer, vegetation_layer):
        rows, cols = biomass_layer.shape
        grid = np.empty((rows, cols), dtype=object)

        for r in range(rows):
            for c in range(cols):
                row = rows - r - 1
                bio_val = float(biomass_layer[row, c])
                veg_val = float(vegetation_layer[row, c])
                fuel_type = self._classify_fuel_type(bio_val, veg_val)
                
                grid[r, c] = {
                    "fuel_type": fuel_type,
                    "biomass": bio_val,
                    "vegetation_ratio": veg_val
                }
                
        return grid

    def get_grid(self):
        return self.grid_data
    
    def get_dimensions(self):
        """Keep in mind, that np.array.shape is (rows, cols)
        and raster images are (width, heigth)"""
        return self.grid_data.shape
    
    def get_crs(self):
        return self.crs
    
    def get_transform(self):
        return self.transform 
