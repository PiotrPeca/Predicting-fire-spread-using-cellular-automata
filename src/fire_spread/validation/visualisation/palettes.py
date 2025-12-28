from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorSpec:
    """Simple color spec container.

    Values should be valid Matplotlib colors.
    """

    background: str = "white"
    grid: str = "#E5E7EB"  # light gray


@dataclass(frozen=True)
class ErrorMapSpec:
    """Defaults for time-difference error maps (sim_toa - real_toa)."""

    cmap: str = "coolwarm"
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True)
class TOASpec:
    """Defaults for Time-of-Arrival (TOA) heatmaps."""

    # Paper-like TOA palette: yellow -> orange -> red
    cmap: str = "YlOrRd"
    vmin: float | None = None
    vmax: float | None = None


DEFAULT_COLORS = ColorSpec()
DEFAULT_TOA = TOASpec()
DEFAULT_ERROR = ErrorMapSpec()
