import numpy as np


def get_valid_twave(
    Ton_Matrix: np.ndarray,
    Tp_Matrix: np.ndarray,
    Toff_Matrix: np.ndarray,
    min_duration_samples: int = 4,
) -> np.ndarray:
    """
    Return a beat-by-lead validity mask for T-wave detections.

    A detection is valid if:
    1) Ton, Tp, and Toff all exist
    2) Ton < Tp < Toff
    3) Toff - Ton >= min_duration_samples
    """
    Ton_Matrix = np.asarray(Ton_Matrix, dtype=int)
    Tp_Matrix = np.asarray(Tp_Matrix, dtype=int)
    Toff_Matrix = np.asarray(Toff_Matrix, dtype=int)

    valid = (Ton_Matrix >= 0) & (Tp_Matrix >= 0) & (Toff_Matrix >= 0)
    valid &= (Ton_Matrix < Tp_Matrix) & (Tp_Matrix < Toff_Matrix)
    valid &= ((Toff_Matrix - Ton_Matrix) >= min_duration_samples)

    return valid


def filter_invalid(mat: np.ndarray, valid_mask: np.ndarray, pad_value: int = -1) -> np.ndarray:
    """
    Replace invalid entries with pad_value.
    """
    out = np.asarray(mat, dtype=int).copy()
    out[~valid_mask] = pad_value
    return out


def median_consensus(times_mat: np.ndarray, fs: int, min_leads: int = 3):
    """
    Compute beat-wise median consensus from a beat-by-lead time matrix.

    Input:
        times_mat: shape (n_beats, n_leads), with -1 marking invalid entries

    Output:
        consensus_sec: shape (n_beats,), NaN if valid leads < min_leads
    """
    x = np.asarray(times_mat, dtype=float)
    x[x < 0] = np.nan

    median_idx = np.nanmedian(x, axis=1)
    n_valid = np.sum(~np.isnan(x), axis=1)

    consensus_sec = np.full(x.shape[0], np.nan, dtype=float)
    ok = n_valid >= min_leads
    consensus_sec[ok] = median_idx[ok] / fs

    return consensus_sec
