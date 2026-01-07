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

    def __init__(self, tiff_path, target_size=None, meters_per_pixel=40, biomass_thresholds=None, vegkvot_thresholds=None):
        """
        :param tiff_path: Path to .tif file
        :param target_size: A tuple (width, height)
        :param meters_per_pixel: Meters per 1 pixel of created grid
        :param biomass_thresholds: Optional dict with 'sparse' and 'normal' keys (e.g., {"sparse": 50, "normal": 150})
        :param vegkvot_thresholds: Optional dict with 'shrub' and 'forest' keys (e.g., {"shrub": 40, "forest": 60})
        """
        self.tiff_path = tiff_path
        self.target_size = target_size
        self.meters_per_pixel = meters_per_pixel
        self.transform = None
        self.crs = None
        
        if biomass_thresholds is not None:
            self.biomass_thresholds = biomass_thresholds
        else:
            self.biomass_thresholds = self.BIOMASS_THRESHOLDS
        
        if vegkvot_thresholds is not None:
            self.vegkvot_thresholds = vegkvot_thresholds
        else:
            self.vegkvot_thresholds = self.VEGKVOT_THRESHOLDS
        
        self.grid_data = self._load_and_process()

    def _load_and_process(self):
        with rasterio.open(self.tiff_path) as dataset:
            src_res = dataset.transform[0]
            target_res = self.meters_per_pixel if self.meters_per_pixel else src_res
            scale = src_res / target_res

            self.crs = dataset.crs

            if self.target_size:
                tgt_w, tgt_h = self.target_size
                src_w, src_h = tgt_w / scale, tgt_h / scale
                
                col_off = (dataset.width - src_w) // 2
                row_off = (dataset.height - src_h) // 2
                
                window = Window(col_off, row_off, src_w, src_h)
                win_transform = get_window_transform(window, dataset.transform)
                self.transform = win_transform * Affine.scale(1/scale, 1/scale)
                out_shape = (int(tgt_h), int(tgt_w))
            else:
                window = None
                self.transform = dataset.transform * Affine.scale(1/scale, 1/scale)
                out_shape = (int(dataset.height * scale), int(dataset.width * scale))

            biomass = dataset.read(5, window=window, out_shape=out_shape, resampling=Resampling.bilinear)
            vegetation = dataset.read(7, window=window, out_shape=out_shape, resampling=Resampling.bilinear)
            
            biomass = np.flip(biomass, axis=0)
            vegetation = np.flip(vegetation, axis=0)
            
            return self._generate_grid_matrix_vectorized(biomass, vegetation)
        
    def _generate_grid_matrix_vectorized(self, biomass_layer, vegetation_layer):
        def classifier_wrapper(b, v):
            return self._classify_fuel_type(b, v)

        v_func = np.vectorize(classifier_wrapper)
        fuel_types = v_func(biomass_layer, vegetation_layer)
        
        rows, cols = biomass_layer.shape
        grid = np.empty((rows, cols), dtype=object)

        for r in range(rows):
            for c in range(cols):
                grid[r, c] = {
                    "fuel_type": fuel_types[r, c],
                    "biomass": float(biomass_layer[r, c]),
                    "vegetation_ratio": float(vegetation_layer[r, c])
                }
        return grid
        
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
                bio_val = float(biomass_layer[r, c])
                veg_val = float(vegetation_layer[r, c])
                fuel_type = self._classify_fuel_type(bio_val, veg_val)
                
                grid[r, c] = {
                    "fuel_type": fuel_type,
                    "biomass": bio_val,
                    "vegetation_ratio": veg_val
                }
                
        return grid

    def get_grid(self, coordinate_system='cartesian'):
        """
        Returns the grid. Default is Cartesian (standard for simulation).
        """
        cs = coordinate_system
        match cs:
            case cs if cs in ('cartesian', 'c'):
                return self.grid_data
            case cs if cs in ('graphics', 'g', 'matrix'):
                return np.flip(self.grid_data, axis=0)
        raise ValueError('coordinate_system invalid value')
    
    
    def get_dimensions(self):
        """Keep in mind, that np.array.shape is (rows, cols)
        and raster images are (width, heigth)"""
        return self.grid_data.shape
    
    def get_crs(self):
        return self.crs
    
    def get_transform(self):
        return self.transform 
    

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Patch

from .cell import FUEL_TYPES


class TerrainVisualizer:
    """Render a categorical terrain grid using matplotlib."""

    def __init__(self, grid: np.ndarray, color_map: dict[str, str] | None = None):
        self.grid = grid
        self.color_map = color_map or self._default_color_map()

    @staticmethod
    def _default_color_map() -> dict[str, str]:
        colors = {}
        for name, fuel in FUEL_TYPES.items():
            color_value = getattr(fuel, "color", None)
            if color_value:
                colors[name] = color_value
        colors.setdefault("water", "#4f93d2")
        colors.setdefault("unknown", "#808080")
        return colors

    def _grid_to_rgb(self) -> tuple[np.ndarray, list[Patch]]:
        rows, cols = self.grid.shape
        rgb = np.zeros((rows, cols, 3), dtype=float)
        legend: dict[str, Patch] = {}

        for row in range(rows):
            for col in range(cols):
                info = self.grid[row, col] or {}
                fuel_type = str(info.get("fuel_type", "unknown"))
                color_hex = self.color_map.get(fuel_type, self.color_map["unknown"])
                rgb[row, col] = mpl_colors.to_rgb(color_hex)
                if fuel_type not in legend:
                    legend[fuel_type] = Patch(facecolor=color_hex, edgecolor="black", label=fuel_type)

        return rgb, list(legend.values())

    def show(self, title: str = "Terrain fuel types") -> None:
        rgb, legend_handles = self._grid_to_rgb()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(rgb, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")

        if legend_handles:
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        fig.tight_layout()
        plt.show()


def main() -> None:
    tiff_filename = '/Users/jakubmikolajczyk/Studia/SSD/Predicting-fire-spread-using-cellular-automata/data/las_20m_resolution.tif'
    tiff_path = Path(tiff_filename).expanduser().resolve()

    if not tiff_path.exists():
        print(f"File not found: {tiff_path}")
        sys.exit(1)

    terrain = Terrain(str(tiff_path))
    grid = terrain.get_grid()

    visualizer = TerrainVisualizer(grid)
    visualizer.show()


if __name__ == "__main__":
    main()

