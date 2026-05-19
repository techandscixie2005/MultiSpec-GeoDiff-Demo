"""Spectrum interpolation, normalization, and preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def interpolate_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    grid_min: float = 400.0,
    grid_max: float = 4000.0,
    grid_size: int = 1800,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate an irregularly-sampled spectrum onto a common grid.

    Parameters
    ----------
    x : ndarray
        Original wavenumber values (must be sorted after handling).
    y : ndarray
        Original intensity values.
    grid_min : float
        Minimum wavenumber (default 400 cm^-1).
    grid_max : float
        Maximum wavenumber (default 4000 cm^-1).
    grid_size : int
        Number of points on the common grid (default 1800).

    Returns
    -------
    common_grid : ndarray, shape (grid_size,)
        Uniformly spaced wavenumber grid.
    interpolated : ndarray, shape (grid_size,)
        Interpolated intensity values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        raise ValueError(f"Too few valid points ({len(x)}) after cleaning.")

    common_grid = np.linspace(grid_min, grid_max, grid_size, dtype=float)

    fn = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    interpolated = fn(common_grid)

    return common_grid, interpolated


def normalize_intensity(y: np.ndarray) -> np.ndarray:
    """Min-max normalize intensity to [0, 1]."""
    y = np.asarray(y, dtype=float)
    y_min, y_max = np.min(y), np.max(y)
    if y_max - y_min < 1e-10:
        return np.zeros_like(y)
    return (y - y_min) / (y_max - y_min)


def extract_peaks(
    x: np.ndarray, y: np.ndarray, prominence_ratio: float = 0.05, max_peaks: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Extract peak positions and intensities from a spectrum.

    Parameters
    ----------
    x : ndarray
        Wavenumber values.
    y : ndarray
        Intensity values (should be normalized).
    prominence_ratio : float
        Minimum prominence as fraction of max intensity.
    max_peaks : int
        Maximum number of peaks to return.

    Returns
    -------
    peak_x : ndarray
        Wavenumber positions of detected peaks.
    peak_y : ndarray
        Intensities at detected peaks.
    """
    prominence = max(0.01, prominence_ratio * float(np.max(y)))
    idx, props = find_peaks(y, prominence=prominence)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(y))])
    values = y[idx]
    order = np.argsort(values)[::-1][:max_peaks]
    idx = idx[order]
    return x[idx].copy(), y[idx].copy()


def preprocess_record(
    record: dict[str, Any],
    grid_min: float = 400.0,
    grid_max: float = 4000.0,
    grid_size: int = 1800,
) -> dict[str, Any]:
    """Preprocess one spectral record: interpolate and normalize.

    Returns a dict with keys:
      - smiles: str
      - wavenumber: ndarray
      - intensity: ndarray
      - peaks_wavenumber: ndarray
      - peaks_intensity: ndarray
      - grid_min, grid_max, grid_size: metadata
    """
    x = np.array(record["value"]["x"], dtype=float)
    y = np.array(record["value"]["y"], dtype=float)
    common_grid, interpolated = interpolate_spectrum(
        x, y, grid_min=grid_min, grid_max=grid_max, grid_size=grid_size
    )
    normalized = normalize_intensity(interpolated)
    peak_x, peak_y = extract_peaks(common_grid, normalized)
    return {
        "smiles": record["smiles"],
        "original_record": record,
        "wavenumber": common_grid,
        "intensity": normalized,
        "peaks_wavenumber": peak_x,
        "peaks_intensity": peak_y,
        "grid_min": grid_min,
        "grid_max": grid_max,
        "grid_size": grid_size,
    }


def preprocess_records(
    records: list[dict[str, Any]],
    grid_min: float = 400.0,
    grid_max: float = 4000.0,
    grid_size: int = 1800,
) -> list[dict[str, Any]]:
    """Preprocess a list of spectral records."""
    return [
        preprocess_record(r, grid_min=grid_min, grid_max=grid_max, grid_size=grid_size)
        for r in records
    ]
