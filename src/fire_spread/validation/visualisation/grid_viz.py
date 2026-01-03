from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np

from .palettes import DEFAULT_TOA, TOASpec


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


def _nanminmax(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, None
    return float(np.nanmin(values)), float(np.nanmax(values))


def _format_metrics(
    metrics: dict[str, Any] | Iterable[tuple[str, Any]] | None,
    *,
    value_format: str,
) -> str | None:
    if not metrics:
        return None

    items = list(metrics.items()) if isinstance(metrics, dict) else list(metrics)
    lines: list[str] = []
    for key, value in items:
        if value is None:
            rendered = "None"
        elif isinstance(value, (int, float, np.floating, np.integer)):
            rendered = value_format.format(float(value))
        else:
            rendered = str(value)
        lines.append(f"{key}={rendered}")
    return "\n".join(lines) if lines else None


class GridVisualizer:
    """Small helper to render validation grids with Matplotlib.

    Designed for *easy* PNG generation and side-by-side comparison.

    Notes:
    - We validate on BURNING (active) cells, so your TOA should represent
      "time when cell first becomes burning", not when it becomes burned.

        Scope:
        - This module only plots already-prepared TOA matrices (real + sim) and
            optional text metrics overlays.
    """

    def __init__(
        self,
        *,
        show_colorbar: bool = True,
        origin: Literal["upper", "lower"] = "upper",
    ) -> None:
        self.show_colorbar = show_colorbar
        self.origin = origin

    def plot_toa(
        self,
        toa: Any,
        *,
        title: str = "Time of arrival",
        spec: TOASpec = DEFAULT_TOA,
        background: Any | None = None,
        background_cmap: str = "gray",
        background_alpha: float = 0.40,
        overlay_alpha: float = 0.80,
        outline: bool = True,
        outline_color: str = "black",
        outline_width: float = 1.2,
        cbar_label: str = "Time since fire start (hours)",
        metrics: dict[str, Any] | list[tuple[str, Any]] | None = None,
        metrics_value_format: str = "{:.2f}",
        metrics_pos: tuple[float, float] = (0.98, 0.98),
        metrics_fontsize: int = 9,
        show_axes: bool = True,
        tick_step: int = 100,
        ax: Any | None = None,
    ) -> Any:
        """Plot a single TOA map.

        - NaNs in TOA are treated as transparent (blank).
        - If outline is True, draws a contour around the finite TOA region.
        """

        toa2d = as_2d_numpy_grid(toa, name="toa").astype(float)

        bg2d: np.ndarray | None = None
        if background is not None:
            bg2d = as_2d_numpy_grid(background, name="background").astype(float)
            if bg2d.shape != toa2d.shape:
                raise ValueError(f"background shape must match TOA. Got {bg2d.shape} vs {toa2d.shape}.")

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

        if outline:
            finite = np.isfinite(toa2d)
            if np.any(finite):
                ax.contour(finite.astype(float), levels=[0.5], colors=[outline_color], linewidths=outline_width)

        metrics_text = _format_metrics(metrics, value_format=metrics_value_format)
        if metrics_text:
            ax.text(
                metrics_pos[0],
                metrics_pos[1],
                metrics_text,
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
            # Paper-like numeric axes: 0, 100, 200, ...
            if tick_step > 0:
                h, w = toa2d.shape
                ax.set_xticks(np.arange(0, w + 1, tick_step))
                ax.set_yticks(np.arange(0, h + 1, tick_step))
                ax.tick_params(axis="both", labelsize=9)

        if self.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cbar.set_label(cbar_label)
        return ax

    def plot_toa_compare(
        self,
        real_toa: Any,
        sim_toa: Any,
        *,
        titles: GridTitles | None = None,
        spec: TOASpec = DEFAULT_TOA,
        background: Any | None = None,
        background_cmap: str = "gray",
        background_alpha: float = 0.40,
        overlay_alpha: float = 0.80,
        outline: bool = True,
        outline_color: str = "black",
        outline_width: float = 1.2,
        cbar_label: str = "Time since fire start (hours)",
        metrics: dict[str, Any] | list[tuple[str, Any]] | None = None,
        metrics_value_format: str = "{:.2f}",
        metrics_pos: tuple[float, float] = (0.98, 0.98),
        metrics_fontsize: int = 9,
        show_axes: bool = True,
        tick_step: int = 100,
        panel_wspace: float | None = None,
    ) -> Any:
        """Plot two TOA matrices side-by-side with a shared colorbar."""

        left2d = as_2d_numpy_grid(real_toa, name="real_toa").astype(float)
        right2d = as_2d_numpy_grid(sim_toa, name="sim_toa").astype(float)
        if left2d.shape != right2d.shape:
            raise ValueError(f"real_toa and sim_toa shapes must match. Got {left2d.shape} vs {right2d.shape}.")

        bg2d: np.ndarray | None = None
        if background is not None:
            bg2d = as_2d_numpy_grid(background, name="background").astype(float)
            if bg2d.shape != left2d.shape:
                raise ValueError(f"background shape must match TOA grids. Got {bg2d.shape} vs {left2d.shape}.")

        titles = titles or GridTitles()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        cmap_obj = plt.get_cmap(spec.cmap).copy()
        cmap_obj.set_bad(alpha=0.0)

        computed_vmin, computed_vmax = _nanminmax(np.stack([left2d, right2d]))
        vmin = spec.vmin if spec.vmin is not None else computed_vmin
        vmax = spec.vmax if spec.vmax is not None else computed_vmax

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

        if outline:
            finite = np.isfinite(left2d)
            if np.any(finite):
                axes[0].contour(finite.astype(float), levels=[0.5], colors=[outline_color], linewidths=outline_width)

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

        if outline:
            finite = np.isfinite(right2d)
            if np.any(finite):
                axes[1].contour(finite.astype(float), levels=[0.5], colors=[outline_color], linewidths=outline_width)

        metrics_text = _format_metrics(metrics, value_format=metrics_value_format)
        if metrics_text:
            axes[1].text(
                metrics_pos[0],
                metrics_pos[1],
                metrics_text,
                transform=axes[1].transAxes,
                ha="right",
                va="top",
                fontsize=metrics_fontsize,
                color="black",
            )

        if not show_axes:
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        else:
            if tick_step > 0:
                h, w = right2d.shape
                axes[1].set_xticks(np.arange(0, w + 1, tick_step))
                axes[1].set_yticks(np.arange(0, h + 1, tick_step))
                axes[1].tick_params(axis="both", labelsize=9)

        if panel_wspace is None:
            panel_wspace = 0.14 if show_axes else 0.06

        if self.show_colorbar:
            fig.subplots_adjust(right=0.88, wspace=panel_wspace)
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(im1, cax=cax)
            if cbar_label:
                cbar.set_label(cbar_label)
        else:
            fig.subplots_adjust(wspace=panel_wspace)

        # Return the figure for saving.
        return fig

    def save(self, fig: Any, path: str, *, dpi: int = 150) -> None:
        """Save a Matplotlib figure to disk."""

        fig.savefig(path, dpi=dpi, bbox_inches="tight")
