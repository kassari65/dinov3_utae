import json
import os
from pathlib import Path
from typing import List

import numpy as np

STATS_PATH = Path(__file__).with_name("stats_per_year.json")
N_FEATURES = 23


def get_indices_of_degree_features():
    """Indices of features that take values in [0,360] and thus will be transformed via sin."""
    return [7, 13, 19]


def get_means_stds_missing_values(training_years: List[int]):
    """Mean/std/missing-value rate of the 23 raw bands for any combination of years.

    Statistics are computed from the data itself (cell 28B of the notebook): per year we store the
    number of valid pixels, the sum and the sum of squares of each band, and merge the years requested here.
    Degree features and the land cover class are not standardized (mean=0, std=1), as in the original code.
    """
    years = [int(y) for y in training_years]
    raw = None
    if STATS_PATH.is_file():
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

    unavailable = years if raw is None else [y for y in years if str(y) not in raw]
    if unavailable:
        if os.environ.get("WSTS_ALLOW_MISSING_STATS") == "1":
            print(f"[utils] Chua co thong ke cho nam {unavailable}; dung gia tri tam (chi dung khi tao HDF5).")
            return (np.zeros(N_FEATURES, dtype=np.float32), np.ones(N_FEATURES, dtype=np.float32),
                    np.zeros(N_FEATURES, dtype=np.float32))
        raise KeyError(
            f"Khong co thong ke cho nam {unavailable} trong {STATS_PATH}. "
            "Chay cell 28B cua notebook de tinh thong ke tu HDF5."
        )

    n = np.zeros(N_FEATURES, dtype=np.float64)
    s = np.zeros(N_FEATURES, dtype=np.float64)
    ss = np.zeros(N_FEATURES, dtype=np.float64)
    total = 0
    for y in years:
        d = raw[str(y)]
        n += np.array(d["n"], dtype=np.float64)
        s += np.array(d["sum"], dtype=np.float64)
        ss += np.array(d["sumsq"], dtype=np.float64)
        total += int(d["total_pixels"])

    mean = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    var = np.divide(ss, n, out=np.zeros_like(ss), where=n > 0) - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    std[std == 0] = 1.0
    missing_values = (1.0 - n / total) if total > 0 else np.ones(N_FEATURES)

    means = mean.astype(np.float32)
    stds = std.astype(np.float32)
    missing_values = np.asarray(missing_values, dtype=np.float32)

    # Zero out means and stds for degree-based features and the categorical land cover type variable
    features_to_not_standardize = get_indices_of_degree_features() + [16]
    means[features_to_not_standardize] = 0
    stds[features_to_not_standardize] = 1

    return means, stds, missing_values
