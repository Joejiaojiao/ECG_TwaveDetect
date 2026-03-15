import numpy as np
from wfdb import processing as wfproc
import io
import contextlib


def _run_quiet(func, *args, **kwargs):
    """Run a function while suppressing stdout and stderr."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return func(*args, **kwargs)


def _deduplicate_close_peaks(
    sig_1d: np.ndarray,
    peaks: np.ndarray,
    fs: int = 100,
    min_rr_ms: int = 450,
) -> np.ndarray:
    """
    Remove peaks that are too close and keep the stronger one.
    """
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size <= 1:
        return peaks

    peaks = np.unique(peaks)
    min_dist = int(round(min_rr_ms / 1000.0 * fs))

    kept = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] < min_dist:
            prev = kept[-1]
            if abs(sig_1d[p]) > abs(sig_1d[prev]):
                kept[-1] = p
        else:
            kept.append(p)

    return np.asarray(kept, dtype=int)


def refine_rpeaks_to_local_peak(
    sig_1d: np.ndarray,
    rpeaks: np.ndarray,
    fs: int = 100,
    search_ms: int = 35,
) -> np.ndarray:
    """
    Refine each R-peak to the local maximum in absolute amplitude.
    """
    sig_1d = np.asarray(sig_1d, dtype=float)
    rpeaks = np.asarray(rpeaks, dtype=int)

    half = int(round((search_ms / 1000.0) * fs))
    L = sig_1d.shape[0]
    out = []

    for p in rpeaks:
        s = max(0, p - half)
        e = min(L, p + half + 1)
        seg = sig_1d[s:e]

        if seg.size == 0:
            out.append(p)
            continue

        k = int(np.argmax(np.abs(seg)))
        out.append(s + k)

    return np.asarray(out, dtype=int)


def refine_rpeaks_per_lead(
    ecg_12xL: np.ndarray,
    rpeaks_ref: np.ndarray,
    fs: int = 100,
    search_ms: int = 35,
    max_shift_ms: int = 60,
) -> np.ndarray:
    """
    Refine lead-wise R-peaks around the reference beat anchors.

    Returns shape (12, n_beats).
    """
    if ecg_12xL.ndim != 2 or ecg_12xL.shape[0] != 12:
        raise ValueError(f"expect (12, L), got {ecg_12xL.shape}")

    rpeaks_ref = np.asarray(rpeaks_ref, dtype=int)
    n = len(rpeaks_ref)
    out = np.full((12, n), -1, dtype=int)

    max_shift = int(round(max_shift_ms / 1000.0 * fs))

    for li in range(12):
        sig = ecg_12xL[li]

        rp = refine_rpeaks_to_local_peak(
            sig,
            rpeaks_ref,
            fs=fs,
            search_ms=search_ms,
        )

        # Limit the deviation from the reference anchors
        too_far = np.abs(rp - rpeaks_ref) > max_shift
        rp[too_far] = rpeaks_ref[too_far]

        # Enforce strict monotonicity across beats
        for k in range(1, len(rp)):
            if rp[k] <= rp[k - 1]:
                rp[k] = max(rp[k - 1] + 1, rpeaks_ref[k])

        out[li] = rp

    return out


def _robust_rr_quality(rpeaks: np.ndarray, fs: int) -> float:
    """
    Score an R-peak sequence using beat count, RR plausibility, and RR stability.
    """
    if rpeaks is None or len(rpeaks) < 2:
        return -1e9

    rpeaks = np.asarray(rpeaks, dtype=int)
    rr = np.diff(rpeaks) / fs

    rr_ok = (rr >= 0.45) & (rr <= 2.00)
    ok_ratio = rr_ok.mean() if len(rr_ok) > 0 else 0.0

    med = np.median(rr)
    mad = np.median(np.abs(rr - med)) + 1e-8
    robust_cv = mad / (med + 1e-8)

    n = len(rpeaks)
    n_score = 1.0 if (6 <= n <= 20) else 0.2

    score = (3.0 * ok_ratio) + (1.0 * n_score) - (1.0 * robust_cv)
    return float(score)


def detect_rpeaks_wfdb(
    sig_1d: np.ndarray,
    fs: int = 100,
    *,
    quiet: bool = True,
    min_rr_ms: int = 450,
) -> np.ndarray:
    """
    Detect R-peaks using WFDB XQRS with a fallback energy-based method.
    """
    sig_1d = np.asarray(sig_1d, dtype=float)

    try:
        if quiet:
            rpeaks = _run_quiet(wfproc.xqrs_detect, sig=sig_1d, fs=fs)
        else:
            rpeaks = wfproc.xqrs_detect(sig=sig_1d, fs=fs)

        rpeaks = np.asarray(rpeaks, dtype=int)
        if rpeaks.size >= 2:
            rpeaks = refine_rpeaks_to_local_peak(sig_1d, rpeaks, fs=fs, search_ms=35)
            rpeaks = _deduplicate_close_peaks(sig_1d, rpeaks, fs=fs, min_rr_ms=min_rr_ms)
            return rpeaks
    except Exception:
        pass

    # Fallback: simple derivative-energy detector
    from scipy.signal import find_peaks

    d = np.diff(sig_1d, prepend=sig_1d[0])
    e = d * d

    win = max(3, int(0.12 * fs))
    kernel = np.ones(win) / win
    env = np.convolve(e, kernel, mode="same")

    thr = np.quantile(env, 0.98) * 0.3
    dist = int(round(min_rr_ms / 1000.0 * fs))
    peaks, _ = find_peaks(env, height=thr, distance=dist)

    peaks = peaks.astype(int)
    peaks = refine_rpeaks_to_local_peak(sig_1d, peaks, fs=fs, search_ms=35)
    peaks = _deduplicate_close_peaks(sig_1d, peaks, fs=fs, min_rr_ms=min_rr_ms)

    return peaks


def select_reference_lead(
    ecg_12xL: np.ndarray,
    fs: int = 100,
    lead_candidates=None,
    *,
    verbose: bool = False,
):
    """
    Select the most reliable lead for reference R-peak detection.

    The score combines beat-count consistency, RR quality, and peak amplitude.
    """
    if ecg_12xL.ndim != 2 or ecg_12xL.shape[0] != 12:
        raise ValueError(f"expect ecg shape (12, L), got {ecg_12xL.shape}")

    if lead_candidates is None:
        lead_candidates = list(range(12))
    else:
        lead_candidates = list(dict.fromkeys(list(lead_candidates)))

    scores = np.full(12, -np.inf, dtype=float)
    cand_rpeaks = {}

    # Step 1: detect R-peaks independently on each lead
    for li in lead_candidates:
        sig = ecg_12xL[li]

        try:
            rp = detect_rpeaks_wfdb(sig, fs=fs, quiet=True, min_rr_ms=450)
            cand_rpeaks[li] = rp
        except Exception as e:
            cand_rpeaks[li] = np.array([], dtype=int)
            if verbose:
                print(f"lead {li}: failed ({e})")

    counts = np.array([len(cand_rpeaks[li]) for li in lead_candidates], dtype=int)
    valid_counts = counts[counts > 0]
    if valid_counts.size == 0:
        raise RuntimeError("Failed to find any valid lead for R-peak detection.")

    median_count = int(np.median(valid_counts))

    # Step 2: score each lead using count consistency, RR quality, and amplitude
    best_idx, best_rpeaks, best_score = None, None, -np.inf

    for li in lead_candidates:
        sig = ecg_12xL[li]
        rpeaks = cand_rpeaks[li]

        if len(rpeaks) < 2:
            continue

        score = _robust_rr_quality(rpeaks, fs)

        count_penalty = 0.8 * abs(len(rpeaks) - median_count)
        score -= count_penalty

        amp = np.median(np.abs(sig[rpeaks])) if len(rpeaks) > 0 else 0.0
        score += 0.2 * float(amp)

        scores[li] = score

        if verbose:
            print(
                f"lead {li}: n_rpeaks={len(rpeaks)}, "
                f"median_count={median_count}, score={score:.3f}"
            )

        if score > best_score:
            best_score = score
            best_idx = li
            best_rpeaks = rpeaks

    if best_idx is None:
        raise RuntimeError("Failed to find a valid reference lead for R-peak detection.")

    return int(best_idx), np.asarray(best_rpeaks, dtype=int), scores
