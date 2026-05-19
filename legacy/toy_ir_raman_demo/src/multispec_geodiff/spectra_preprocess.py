from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter


@dataclass
class ProcessedSpectrum:
    modality: str
    wavenumber: np.ndarray
    intensity: np.ndarray
    peaks: np.ndarray
    peak_intensities: np.ndarray
    metadata: dict


def make_common_grid(min_wavenumber: float, max_wavenumber: float, step: float) -> np.ndarray:
    return np.arange(min_wavenumber, max_wavenumber + step, step, dtype=float)


def normalize_intensity(values: Iterable[float], method: str = "max") -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if method == "max":
        denom = np.max(np.abs(arr))
        return arr / denom if denom > 0 else arr
    if method == "sum":
        denom = np.sum(np.abs(arr))
        return arr / denom if denom > 0 else arr
    if method == "zscore":
        std = np.std(arr)
        return (arr - np.mean(arr)) / std if std > 0 else arr - np.mean(arr)
    raise ValueError(f"Unsupported normalization method: {method}")


def smooth_intensity(values: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 7:
        return values
    window = min(window_length, values.size if values.size % 2 == 1 else values.size - 1)
    window = max(window, polyorder + 3)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return values
    return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")


def resample_spectrum(df: pd.DataFrame, common_grid: np.ndarray) -> pd.DataFrame:
    fn = interp1d(
        df["wavenumber"].to_numpy(dtype=float),
        df["intensity"].to_numpy(dtype=float),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return pd.DataFrame({"wavenumber": common_grid, "intensity": fn(common_grid)})


def extract_peaks(wavenumber: np.ndarray, intensity: np.ndarray, threshold: float = 0.35, max_peaks: int = 12) -> tuple[np.ndarray, np.ndarray]:
    if intensity.size == 0:
        return np.array([]), np.array([])
    prominence = max(0.05, threshold * float(np.max(intensity)))
    idx, props = find_peaks(intensity, prominence=prominence)
    if idx.size == 0:
        idx = np.array([int(np.argmax(intensity))])
    values = intensity[idx]
    order = np.argsort(values)[::-1][:max_peaks]
    idx = idx[order]
    return wavenumber[idx], intensity[idx]


def preprocess_spectrum(
    df: pd.DataFrame,
    modality: str,
    common_grid: np.ndarray,
    normalize: str = "max",
    smoothing_window: int = 11,
    smoothing_polyorder: int = 2,
    peak_threshold: float = 0.35,
) -> ProcessedSpectrum:
    resampled = resample_spectrum(df, common_grid)
    smoothed = smooth_intensity(resampled["intensity"].to_numpy(dtype=float), smoothing_window, smoothing_polyorder)
    normalized = normalize_intensity(smoothed, method=normalize)
    peaks, peak_intensities = extract_peaks(common_grid, normalized, threshold=peak_threshold)
    metadata = {
        "normalize": normalize,
        "smoothing_window": smoothing_window,
        "smoothing_polyorder": smoothing_polyorder,
        "peak_threshold": peak_threshold,
        "num_peaks": int(len(peaks)),
    }
    return ProcessedSpectrum(
        modality=modality,
        wavenumber=common_grid.copy(),
        intensity=normalized,
        peaks=peaks,
        peak_intensities=peak_intensities,
        metadata=metadata,
    )


def preprocess_optional_spectrum(df: pd.DataFrame | None, modality: str, common_grid: np.ndarray, **kwargs) -> ProcessedSpectrum | None:
    if df is None:
        return None
    return preprocess_spectrum(df=df, modality=modality, common_grid=common_grid, **kwargs)
