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
class TOASpec:
    """Defaults for Time-of-Arrival (TOA) heatmaps."""

    # Matches colleague's reference style (IgnitionProcessor.plot_results)
    cmap: str = "plasma_r"
    vmin: float | None = None
    vmax: float | None = None


DEFAULT_COLORS = ColorSpec()
DEFAULT_TOA = TOASpec()
