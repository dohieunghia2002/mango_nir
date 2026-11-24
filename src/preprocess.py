"""M1 - spectral pre-treatment: optional SNV, Savitzky-Golay derivative on the full spectrum, then trim to a window."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class PreprocessConfig:
    # Defaults follow Anderson et al. 2020: SG second derivative, 2nd-order polynomial, 17 points, 684-990 nm.
    deriv: int = 2
    window: int = 17
    polyorder: int = 2
    snv: bool = False
    wl_min: int = 684
    wl_max: int = 990


def snv(X, ref_mask=None, ddof=0):
    """Standard normal variate per spectrum, with mean and SD taken over the columns in ref_mask (all by default)."""
    R = X if ref_mask is None else X[:, ref_mask]
    return (X - R.mean(axis=1, keepdims=True)) / R.std(axis=1, ddof=ddof, keepdims=True)


def preprocess(X, wl, cfg=PreprocessConfig()):
    """Return (X_trimmed, wl_trimmed). The SG filter runs on the full spectrum before trimming, so the window edges
    are not distorted by the filter boundary."""
    X = np.asarray(X, dtype=float)
    wl = np.asarray(wl)
    keep = (wl >= cfg.wl_min) & (wl <= cfg.wl_max)
    if cfg.snv:
        # SNV statistics over the model window only: the file spans 285-1200 nm, wider than the F750's 300-1100 nm.
        X = snv(X, keep)
    X = savgol_filter(X, cfg.window, cfg.polyorder, deriv=cfg.deriv, axis=1)
    return X[:, keep], wl[keep]


AUGMENT_BLOCKS = ('raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2')
# SNV statistics of the 'AT' pretreatment. Walsh's published 'AT' input files (mishra&passos_2021_pretreatment_adjusted.csv,
# Mendeley Data doi:10.17632/mc9jshwb38.1) are reproduced to rounding error only with the mean and SD (ddof=1) taken over
# 402-1098 nm; the reason for this range is not stated in their code.
AT_SNV_RANGE = (402, 1098)


def augment(X, wl, trim_first=False, wl_min=684, wl_max=990, window=13, polyorder=2, at_snv_range=AT_SNV_RANGE):
    """Six pretreatments of absorbance concatenated (Mishra & Passos 2021, as described by Walsh et al. 2024, 2.1):
    nil, SNV, SG 1st and 2nd derivative (13 points, 2nd order), and SNV followed by each derivative -> 6 x 103 = 618.
    trim_first=True trims to the window and then pretreats ('TA'): SNV statistics over the 103 trimmed variables.
    False pretreats the full spectrum and then trims ('AT'): SNV statistics over at_snv_range nm.
    SNV uses the sample SD (ddof=1), as Walsh's input files do. Both variants match those files to rounding error."""
    X = np.asarray(X, dtype=float)
    wl = np.asarray(wl)
    keep = (wl >= wl_min) & (wl <= wl_max)
    if trim_first:
        X = X[:, keep]
        keep = np.ones(X.shape[1], dtype=bool)
        ref = None
    else:
        ref = (wl >= at_snv_range[0]) & (wl <= at_snv_range[1])
    S = snv(X, ref, ddof=1)
    blocks = [X, S] + [savgol_filter(B, window, polyorder, deriv=d, axis=1) for B in (X, S) for d in (1, 2)]
    return np.hstack([b[:, keep] for b in blocks])
