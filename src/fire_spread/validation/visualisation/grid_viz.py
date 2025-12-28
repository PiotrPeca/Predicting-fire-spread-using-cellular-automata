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
        background: Any | None = None,
        background_cmap: str = "gray",
        background_alpha: float = 0.55,
        overlay_alpha: float = 0.90,
        mask: Any | None = None,
        outline_mask: bool = True,
        outline_color: str = "#00B5FF",
        outline_width: float = 1.5,
        cbar_label: str = "TOA (minutes from t0)",
        show_axes: bool = False,
        tick_step: int = 100,
        ax: Any | None = None,
    ) -> Any:
        """Plot a single TOA map with an optional mask boundary.

        TOA is interpreted as: time (e.g. minutes from t0) when a cell FIRST
        becomes BURNING.

        - If mask is provided, values outside the mask are set to NaN (blank).
        - If outline_mask is True, draws a contour line around the mask.
        """

        toa2d = as_2d_numpy_grid(toa, name="toa").astype(float)

        bg2d: np.ndarray | None = None
        if background is not None:
            bg2d = as_2d_numpy_grid(background, name="background").astype(float)
            if bg2d.shape != toa2d.shape:
                raise ValueError(f"background shape must match TOA. Got {bg2d.shape} vs {toa2d.shape}.")

        mask2d: np.ndarray | None = None
        if mask is not None:
            mask2d = as_2d_numpy_grid(mask, name="mask").astype(bool)
            if mask2d.shape != toa2d.shape:
                raise ValueError(f"mask shape must match TOA. Got {mask2d.shape} vs {toa2d.shape}.")
            toa2d = np.where(mask2d, toa2d, np.nan)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 5))

        # 1) Background layer (e.g., terrain types) in grayscale.
        if bg2d is not None:
            ax.imshow(bg2d, cmap=background_cmap, origin=self.origin, alpha=background_alpha)

        # 2) TOA overlay: make NaN fully transparent so background stays visible.
        cmap = plt.get_cmap(spec.cmap).copy()
        cmap.set_bad(alpha=0.0)

        im = ax.imshow(
            toa2d,
            cmap=cmap,
            vmin=spec.vmin,
            vmax=spec.vmax,
            origin=self.origin,
            alpha=overlay_alpha,
        )
        ax.set_title(title)

        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Paper-like numeric axes: 0, 100, 200, ...
            if tick_step > 0:
                h, w = toa2d.shape
                ax.set_xticks(np.arange(0, w + 1, tick_step))
                ax.set_yticks(np.arange(0, h + 1, tick_step))
                ax.tick_params(axis="both", labelsize=9)

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
        background: Any | None = None,
        background_cmap: str = "gray",
        background_alpha: float = 0.55,
        overlay_alpha: float = 0.90,
        show_axes: bool = False,
        tick_step: int = 100,
        panel_wspace: float | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        cbar_label: str | None = None,
    ) -> Any:
        """Plot two grids side-by-side (for comparison)."""

        left2d = as_2d_numpy_grid(left, name="left")
        right2d = as_2d_numpy_grid(right, name="right")

        bg2d: np.ndarray | None = None
        if background is not None:
            bg2d = as_2d_numpy_grid(background, name="background").astype(float)
            if bg2d.shape != left2d.shape or bg2d.shape != right2d.shape:
                raise ValueError(
                    "background shape must match left/right. "
                    f"Got {bg2d.shape} vs left={left2d.shape} right={right2d.shape}."
                )

        titles = titles or GridTitles()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # If grids contain NaNs, make them render as transparent (blank).
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)

        if bg2d is not None:
            axes[0].imshow(bg2d, cmap=background_cmap, origin=self.origin, alpha=background_alpha)
        im0 = axes[0].imshow(
            left2d,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            origin=self.origin,
            alpha=overlay_alpha,
        )
        axes[0].set_title(titles.left)

        if not show_axes:
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        else:
            if tick_step > 0:
                h, w = left2d.shape
                axes[0].set_xticks(np.arange(0, w + 1, tick_step))
                axes[0].set_yticks(np.arange(0, h + 1, tick_step))
                axes[0].tick_params(axis="both", labelsize=9)

        if bg2d is not None:
            axes[1].imshow(bg2d, cmap=background_cmap, origin=self.origin, alpha=background_alpha)
        im1 = axes[1].imshow(
            right2d,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            origin=self.origin,
            alpha=overlay_alpha,
        )
        axes[1].set_title(titles.right)

        if not show_axes:
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        else:
            if tick_step > 0:
                h, w = right2d.shape
                axes[1].set_xticks(np.arange(0, w + 1, tick_step))
                axes[1].set_yticks(np.arange(0, h + 1, tick_step))
                axes[1].tick_params(axis="both", labelsize=9)

        if self.show_colorbar:
            # Put the colorbar in its own axis fully to the right.
            # This avoids overlap with the right subplot.
            # Increase spacing when axes/ticks are shown to avoid label overlap.
            if panel_wspace is None:
                panel_wspace = 0.14 if show_axes else 0.06
            fig.subplots_adjust(right=0.88, wspace=panel_wspace)
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(im1, cax=cax)
            if cbar_label:
                cbar.set_label(cbar_label)
        else:
            if panel_wspace is None:
                panel_wspace = 0.14 if show_axes else 0.06
            fig.subplots_adjust(wspace=panel_wspace)

        return fig

    # -------- validation-specific plots --------

    def plot_error_map(
        self,
        sim_toa: Any,
        real_toa: Any,
        *,
        spec: ErrorMapSpec = DEFAULT_ERROR,
        title: str = "Error map (sim_toa - real_toa)",
        background: Any | None = None,
        background_cmap: str = "gray",
        background_alpha: float = 0.55,
        overlay_alpha: float = 0.90,
        mask: Any | None = None,
        metrics: dict[str, Any] | list[tuple[str, Any]] | None = None,
        metrics_value_format: str = "{:.1f}",
        metrics_pos: tuple[float, float] = (0.98, 0.98),
        metrics_fontsize: int = 9,
        cbar_label: str = "Δt (minutes)",
        show_axes: bool = False,
        tick_step: int = 100,
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

        bg2d: np.ndarray | None = None
        if background is not None:
            bg2d = as_2d_numpy_grid(background, name="background").astype(float)
            if bg2d.shape != diff.shape:
                raise ValueError(f"background shape must match grids. Got {bg2d.shape} vs {diff.shape}.")

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        if bg2d is not None:
            ax.imshow(bg2d, cmap=background_cmap, origin=self.origin, alpha=background_alpha)

        cmap_obj = plt.get_cmap(spec.cmap).copy()
        cmap_obj.set_bad(alpha=0.0)
        im = ax.imshow(
            diff,
            cmap=cmap_obj,
            vmin=spec.vmin,
            vmax=spec.vmax,
            origin=self.origin,
            alpha=overlay_alpha,
        )
        ax.set_title(title)

        if metrics:
            if isinstance(metrics, dict):
                items = list(metrics.items())
            else:
                items = list(metrics)

            lines: list[str] = []
            for key, value in items:
                if value is None:
                    rendered = "None"
                elif isinstance(value, (int, float, np.floating, np.integer)):
                    rendered = metrics_value_format.format(float(value))
                else:
                    rendered = str(value)
                lines.append(f"{key}={rendered}")

            ax.text(
                metrics_pos[0],
                metrics_pos[1],
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=metrics_fontsize,
                color="black",
            )

        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            if tick_step > 0:
                h, w = diff.shape
                ax.set_xticks(np.arange(0, w + 1, tick_step))
                ax.set_yticks(np.arange(0, h + 1, tick_step))
                ax.tick_params(axis="both", labelsize=9)

        if self.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cbar.set_label(cbar_label)
        fig.tight_layout()
        return fig

    def save(self, fig: Any, path: str, *, dpi: int = 150) -> None:
        """Save a Matplotlib figure to disk."""

        fig.savefig(path, dpi=dpi, bbox_inches="tight")
