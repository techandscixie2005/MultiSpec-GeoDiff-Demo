"""Tests for spectra_preprocess module."""

import numpy as np
import pytest


def test_interpolate_spectrum():
    from spectra_preprocess import interpolate_spectrum
    x = np.array([400, 1000, 2000, 3000, 4000], dtype=float)
    y = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=float)
    grid, interp = interpolate_spectrum(x, y, grid_min=400, grid_max=4000, grid_size=361)
    assert len(grid) == 361
    assert len(interp) == 361
    assert grid[0] == 400.0
    assert grid[-1] == 4000.0
    assert np.all(np.isfinite(interp))


def test_interpolate_spectrum_irregular():
    """Handle out-of-order x values."""
    from spectra_preprocess import interpolate_spectrum
    x = np.array([3000, 1000, 4000, 400, 2000], dtype=float)
    y = np.array([0.5, 0.0, 0.0, 0.0, 1.0], dtype=float)
    grid, interp = interpolate_spectrum(x, y, grid_min=400, grid_max=4000, grid_size=361)
    assert np.all(np.isfinite(interp))


def test_interpolate_too_few_points():
    from spectra_preprocess import interpolate_spectrum
    with pytest.raises(ValueError):
        interpolate_spectrum(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_normalize_intensity():
    from spectra_preprocess import normalize_intensity
    y = np.array([0.0, 0.5, 1.0, 0.0], dtype=float)
    normalized = normalize_intensity(y)
    assert np.min(normalized) == 0.0
    assert np.max(normalized) == 1.0
    assert len(normalized) == 4


def test_normalize_constant():
    from spectra_preprocess import normalize_intensity
    y = np.ones(10)
    normalized = normalize_intensity(y)
    assert np.all(normalized == 0.0)


def test_extract_peaks():
    from spectra_preprocess import extract_peaks
    x = np.linspace(400, 4000, 361)
    y = np.exp(-((x - 1500) ** 2) / (2 * 100 ** 2))
    peaks_x, peaks_y = extract_peaks(x, y)
    assert len(peaks_x) >= 1
    assert abs(peaks_x[0] - 1500) < 50


def test_preprocess_record():
    from spectra_preprocess import preprocess_record
    record = {
        "smiles": "CCO",
        "value": {
            "x": [400, 1000, 2000, 3000, 4000],
            "y": [0.0, 0.5, 1.0, 0.5, 0.0],
        },
    }
    result = preprocess_record(record)
    assert result["smiles"] == "CCO"
    assert len(result["wavenumber"]) == 1800
    assert len(result["intensity"]) == 1800
    assert np.max(result["intensity"]) == 1.0
    assert "peaks_wavenumber" in result
