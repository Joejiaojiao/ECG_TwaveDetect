import numpy as np
from typing import Dict, List, Tuple


def compute_t_windows(
    rpeaks: np.ndarray,
    fs: int = 100,
    t_start_ms: float = 100.0,
    t_end_max_ms: float = 600.0,
    t_end_rr_ratio: float = 0.70,
) -> List[Tuple[int, int]]:
    """
    Build beat-wise T-wave search windows [start, end) between adjacent R-peaks.

    Input:
        rpeaks: shape (K,), sample indices in ascending order

    Return:
        list of (start_idx, end_idx) for beats 0..K-2
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    windows = []

    offset_start = int(round((t_start_ms / 1000.0) * fs))
    offset_end_max = int(round((t_end_max_ms / 1000.0) * fs))

    for i in range(len(rpeaks) - 1):
        r0 = rpeaks[i]
        r1 = rpeaks[i + 1]
        rr = r1 - r0
        if rr <= 0:
            continue

        start = r0 + offset_start
        end1 = r0 + offset_end_max
        end2 = r0 + int(round(t_end_rr_ratio * rr))
        end = min(end1, end2)

        # Require a minimum window length of 80 ms
        if end - start < int(0.08 * fs):
            continue

        windows.append((start, end))

    return windows


def detect_tpeaks_per_lead(
    ecg_filt_12xL: np.ndarray,
    rpeaks: np.ndarray,
    fs: int = 100,
    leads: List[int] = None,
    use_abs: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Detect beat-wise T-peaks for each lead.

    Input:
        ecg_filt_12xL: shape (12, L), preferably filtered but not standardized
        rpeaks: shape (K,)
        leads: list of leads to process; None means all 12 leads
        use_abs: if True, detect the maximum in absolute amplitude

    Return:
        tpeaks_by_lead: dict {lead_idx: np.ndarray of shape (n_beats,)}
        windows: list of beat-wise T-wave search windows
    """
    if ecg_filt_12xL.ndim != 2 or ecg_filt_12xL.shape[0] != 12:
        raise ValueError(f"expect ecg shape (12,L), got {ecg_filt_12xL.shape}")

    L = ecg_filt_12xL.shape[1]
    rpeaks = np.asarray(rpeaks, dtype=int)

    if leads is None:
        leads = list(range(12))

    windows = compute_t_windows(rpeaks, fs=fs)
    if len(windows) == 0:
        raise RuntimeError("No valid T windows computed from rpeaks. Check rpeaks/fs.")

    tpeaks_by_lead: Dict[int, np.ndarray] = {}

    for li in leads:
        sig = ecg_filt_12xL[li]
        tpeaks = []

        for (start, end) in windows:
            s = max(0, start)
            e = min(L, end)
            if e <= s + 1:
                tpeaks.append(-1)
                continue

            seg = sig[s:e]
            if use_abs:
                local = int(np.argmax(np.abs(seg)))
            else:
                local = int(np.argmax(seg))

            tpeaks.append(s + local)

        tpeaks_by_lead[li] = np.asarray(tpeaks, dtype=int)

    return tpeaks_by_lead, windows


def _first_run_below(x: np.ndarray, thr: float, consec: int) -> int:
    """
    Return the first index i such that x[i:i+consec] is entirely <= thr.
    Return -1 if no such run is found.
    """
    if consec <= 1:
        idx = np.where(x <= thr)[0]
        return int(idx[0]) if idx.size else -1
    for i in range(0, len(x) - consec + 1):
        if np.all(x[i:i + consec] <= thr):
            return i
    return -1


def detect_tonoff_given_tpeak(
    ecg_filt_12xL: np.ndarray,
    tpeaks_by_lead: dict,
    windows: list,
    fs: int,
    leads=None,
    baseline_ms: int = 80,
    frac: float = 0.10,
    consec_ms: int = 20,
    use_abs: bool = True,
):
    """
    Detect T-wave onset and offset given T-peak locations.

    Input:
        ecg_filt_12xL: shape (12, L)
        tpeaks_by_lead: dict[lead] -> (n_beats,) int, with -1 for invalid peaks
        windows: list of (start, end) sample indices
        fs: sampling rate

    Output:
        ton_by_lead: dict[lead] -> (n_beats,) int
        toff_by_lead: dict[lead] -> (n_beats,) int
    """
    if leads is None:
        leads = list(range(ecg_filt_12xL.shape[0]))

    n_beats = len(windows)
    ton_by_lead = {}
    toff_by_lead = {}

    baseline_len = max(1, int(baseline_ms * fs / 1000.0))
    consec = max(1, int(consec_ms * fs / 1000.0))

    for li in leads:
        sig = ecg_filt_12xL[li]
        tp = np.asarray(tpeaks_by_lead[li], dtype=int)
        ton = np.full(n_beats, -1, dtype=int)
        toff = np.full(n_beats, -1, dtype=int)

        for b, (s, e) in enumerate(windows):
            if b >= tp.shape[0]:
                continue
            tpk = tp[b]
            if tpk < 0:
                continue

            s0 = max(0, int(s))
            e0 = min(len(sig) - 1, int(e))
            tpk = int(np.clip(tpk, s0, e0))

            # Estimate baseline from the segment before the window
            bs0 = max(0, s0 - baseline_len)
            baseline_seg = sig[bs0:s0] if s0 > bs0 else sig[max(0, s0 - 1):s0 + 1]
            baseline = float(np.median(baseline_seg))
            mad = float(np.median(np.abs(baseline_seg - baseline))) + 1e-12
            sigma = 1.4826 * mad

            # Build an adaptive threshold from peak amplitude and baseline noise
            A = abs(sig[tpk] - baseline)
            if A < 1e-6:
                continue

            thr = max(frac * A, 2.5 * sigma)

            if use_abs:
                dist = np.abs(sig[s0:e0 + 1] - baseline)
            else:
                dist = sig[s0:e0 + 1] - baseline

            # Onset: search left from T-peak
            left = dist[: (tpk - s0 + 1)]
            left_rev = left[::-1]
            i0 = _first_run_below(left_rev, thr, consec)
            if i0 != -1:
                ton[b] = tpk - i0

            # Offset: search right from T-peak
            right = dist[(tpk - s0):]
            i1 = _first_run_below(right, thr, consec)
            if i1 != -1:
                toff[b] = tpk + i1

        ton_by_lead[li] = ton
        toff_by_lead[li] = toff

    return ton_by_lead, toff_by_lead
