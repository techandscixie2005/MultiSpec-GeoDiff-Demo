from __future__ import annotations

import numpy as np
import pytest

from multispec_geodiff.spectra_preprocess import normalize_intensity


def test_normalize_intensity_scales_by_max_absolute_value() -> None:
    normalized = normalize_intensity([1.0, -2.0, 0.5], method="max")

    assert np.allclose(normalized, [0.5, -1.0, 0.25])


def test_normalize_intensity_returns_zero_mean_unit_std_for_zscore() -> None:
    normalized = normalize_intensity([1.0, 2.0, 3.0], method="zscore")

    assert np.isclose(np.mean(normalized), 0.0)
    assert np.isclose(np.std(normalized), 1.0)


def test_normalize_intensity_rejects_unsupported_method() -> None:
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        normalize_intensity([1.0, 2.0], method="median")
