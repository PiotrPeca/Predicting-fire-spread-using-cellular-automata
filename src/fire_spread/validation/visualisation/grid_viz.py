from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from .palettes import DEFAULT_ERROR, DEFAULT_TOA, ErrorMapSpec, TOASpec


def as_2d_numpy_grid(grid: Any, *, name: str = "grid") -> np.ndarray:
    """Coerce input to a 2D numpy array.

    - Accepts list-likes or numpy arrays.
    - Returns a view/copy as needed.
    """

    array = np.asarray(grid)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array (H x W). Got shape={array.shape}.")
    return array


@dataclass(frozen=True)
class GridTitles:
    left: str = "Real"
    right: str = "Sim"
    diff: str = "Sim - Real"


class GridVisualizer:
    """Small helper to render validation grids with Matplotlib.

    Designed for *easy* PNG generation and side-by-side comparison.

    Notes:
    - We validate on BURNING (active) cells, so your TOA should represent
      "time when cell first becomes burning", not when it becomes burned.
    """

    def __init__(
        self,
        *,
        show_colorbar: bool = True,
        origin: Literal["upper", "lower"] = "upper",
    ) -> None:
        self.show_colorbar = show_colorbar
        self.origin = origin

    # -------- basic plots --------

    def plot_grid(
        self,
        grid: Any,
        *,
        title: str | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        cbar_label: str | None = None,
        ax: Any | None = None,
    ) -> Any:
        """Plot a single grid (e.g., burning mask or TOA map)."""

        grid2d = as_2d_numpy_grid(grid)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 5))

        im = ax.imshow(grid2d, cmap=cmap, vmin=vmin, vmax=vmax, origin=self.origin)
        if title:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        if self.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cbar.set_label(cbar_label)
        return ax

    def plot_toa(
        self,
        toa: Any,
        *,
        title: str = "TOA (time of arrival)",
        spec: TOASpec = DEFAULT_TOA,
        mask: Any | None = None,
        outline_mask: bool = True,
        outline_color: str = "#00B5FF",
        outline_width: float = 1.5,
        cbar_label: str = "TOA (minutes from t0)",
        ax: Any | None = None,
    ) -> Any:
        """Plot a single TOA map with an optional mask boundary.

        TOA is interpreted as: time (e.g. minutes from t0) when a cell FIRST
        becomes BURNING.

        - If mask is provided, values outside the mask are set to NaN (blank).
        - If outline_mask is True, draws a contour line around the mask.
        """

        toa2d = as_2d_numpy_grid(toa, name="toa").astype(float)

        mask2d: np.ndarray | None = None
        if mask is not None:
            mask2d = as_2d_numpy_grid(mask, name="mask").astype(bool)
            if mask2d.shape != toa2d.shape:
                raise ValueError(f"mask shape must match TOA. Got {mask2d.shape} vs {toa2d.shape}.")
            toa2d = np.where(mask2d, toa2d, np.nan)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 5))

        im = ax.imshow(
            toa2d,
            cmap=spec.cmap,
            vmin=spec.vmin,
            vmax=spec.vmax,
            origin=self.origin,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        if outline_mask and mask2d is not None:
            # Draw the boundary of the analysis mask (like a perimeter line).
            ax.contour(mask2d.astype(float), levels=[0.5], colors=[outline_color], linewidths=outline_width)

        if self.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cbar.set_label(cbar_label)
        return ax

    def plot_two_grids(
        self,
        left: Any,
        right: Any,
        *,
        titles: GridTitles | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        cbar_label: str | None = None,
    ) -> Any:
        """Plot two grids side-by-side (for comparison)."""

        left2d = as_2d_numpy_grid(left, name="left")
        right2d = as_2d_numpy_grid(right, name="right")

        titles = titles or GridTitles()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axes[0].imshow(left2d, cmap=cmap, vmin=vmin, vmax=vmax, origin=self.origin)
        axes[0].set_title(titles.left)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        im1 = axes[1].imshow(right2d, cmap=cmap, vmin=vmin, vmax=vmax, origin=self.origin)
        axes[1].set_title(titles.right)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        if self.show_colorbar:
            # Put the colorbar in its own axis fully to the right.
            # This avoids overlap with the right subplot.
            fig.subplots_adjust(right=0.88, wspace=0.05)
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(im1, cax=cax)
            if cbar_label:
                cbar.set_label(cbar_label)
        else:
            fig.subplots_adjust(wspace=0.05)

        return fig

    # -------- validation-specific plots --------

    def plot_error_map(
        self,
        sim_toa: Any,
        real_toa: Any,
        *,
        spec: ErrorMapSpec = DEFAULT_ERROR,
        title: str = "Error map (sim_toa - real_toa)",
        mask: Any | None = None,
        cbar_label: str = "Δt (minutes)",
    ) -> Any:
        """Plot error map Δt = sim_toa - real_toa.

        If mask is provided (bool 2D), values outside mask are set to NaN.
        """

        sim2d = as_2d_numpy_grid(sim_toa, name="sim_toa").astype(float)
        real2d = as_2d_numpy_grid(real_toa, name="real_toa").astype(float)

        if sim2d.shape != real2d.shape:
            raise ValueError(f"sim_toa and real_toa shapes must match. Got {sim2d.shape} vs {real2d.shape}.")

        diff = sim2d - real2d

        if mask is not None:
            mask2d = as_2d_numpy_grid(mask, name="mask")
            if mask2d.shape != diff.shape:
                raise ValueError(f"mask shape must match grids. Got {mask2d.shape} vs {diff.shape}.")
            diff = np.where(mask2d.astype(bool), diff, np.nan)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(diff, cmap=spec.cmap, vmin=spec.vmin, vmax=spec.vmax, origin=self.origin)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        if self.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cbar.set_label(cbar_label)
        fig.tight_layout()
        return fig

    def save(self, fig: Any, path: str, *, dpi: int = 150) -> None:
        """Save a Matplotlib figure to disk."""

        fig.savefig(path, dpi=dpi, bbox_inches="tight")
