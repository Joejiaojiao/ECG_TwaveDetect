# SpecialTask/Pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .Preprocess import TwavePreprocessor
from .Twave import detect_tpeaks_per_lead, detect_tonoff_given_tpeak
from .Consensus import get_valid_twave, filter_invalid, median_consensus
from .Visual import build_tpeak_matrix
from .Rpeak import (
    select_reference_lead,
    refine_rpeaks_to_local_peak,
    refine_rpeaks_per_lead,
)


def _ensure_12xL(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"ECG must be 2D, got shape={x.shape}")
    if x.shape[0] == 12:
        return x
    if x.shape[1] == 12:
        return x.T
    raise ValueError(f"ECG must be 12xL or Lx12, got shape={x.shape}")


def _to_sec(arr_idx: np.ndarray, fs: int) -> np.ndarray:
    arr = arr_idx.astype(float)
    arr[arr < 0] = np.nan
    return arr / fs


def detect_twaves_12lead(ecg: np.ndarray, fs: int = 100):

    # ---------- 1 Preprocessing ----------
    x12 = _ensure_12xL(ecg)

    prep = TwavePreprocessor(
        fs=fs,
        k1=21,
        k2=61,
        lp_cut=35.0,
        lp_taps=101,
    )

    ecg_filt, ecg_std = prep(
        x12,
        return_standardized_for_rpeak=True,
        out_layout="12xL",
    )

    # ---------- 2 Reference lead and R-peaks ----------
    ref_idx, rpeaks_ref, scores = select_reference_lead(ecg_std, fs=fs)

    rpeaks_ref = refine_rpeaks_to_local_peak(
        ecg_std[ref_idx],
        rpeaks_ref,
        fs=fs,
        search_ms=50,
    )

    rpeaks_by_lead = refine_rpeaks_per_lead(
        ecg_std, rpeaks_ref, fs=fs, search_ms=80
    )

    # ---------- 3 T-peak detection ----------
    tpeaks_by_lead, windows = detect_tpeaks_per_lead(
        ecg_filt,
        rpeaks_ref,
        fs=fs,
        leads=None,
        use_abs=True,
    )

    # ---------- 4 Ton / Toff detection ----------
    ton_by_lead, toff_by_lead = detect_tonoff_given_tpeak(
        ecg_filt,
        tpeaks_by_lead,
        windows,
        fs=fs,
        leads=None,
        frac=0.10,
        consec_ms=10,
        use_abs=True,
    )

    # ---------- 5 Consensus ----------
    n_beats = len(windows)

    TpMat = build_tpeak_matrix(
        tpeaks_by_lead,
        n_beats=n_beats,
        n_leads=12,
        pad_value=-1,
        verbose=False,
    )

    TonMat = build_tpeak_matrix(
        ton_by_lead,
        n_beats=n_beats,
        n_leads=12,
        pad_value=-1,
        verbose=False,
    )

    ToffMat = build_tpeak_matrix(
        toff_by_lead,
        n_beats=n_beats,
        n_leads=12,
        pad_value=-1,
        verbose=False,
    )

    # Convert Ton/Toff from (n_beats, 12) to (12, n_beats)
    TonL = TonMat.T
    ToffL = ToffMat.T

    # Stack into shape (12, n_beats, 2)
    TonToff_pair = np.stack([TonL, ToffL], axis=-1)
    TonToff_pair_sec = TonToff_pair.astype(float)
    TonToff_pair_sec[TonToff_pair_sec < 0] = np.nan
    TonToff_pair_sec /= fs

    valid_mask = get_valid_twave(
        TonMat,
        TpMat,
        ToffMat,
        min_duration_samples=4,   # 40 ms @ 100 Hz
    )

    TonMat_valid = filter_invalid(TonMat, valid_mask, pad_value=-1)
    TpMat_valid = filter_invalid(TpMat, valid_mask, pad_value=-1)
    ToffMat_valid = filter_invalid(ToffMat, valid_mask, pad_value=-1)

    ton_cons_sec = median_consensus(TonMat_valid, fs=fs, min_leads=3)
    tp_cons_sec = median_consensus(TpMat_valid, fs=fs, min_leads=3)
    toff_cons_sec = median_consensus(ToffMat_valid, fs=fs, min_leads=3)

    # ---------- Return ----------
    return {
        "fs": fs,
        "ecg_filt": ecg_filt,

        "ref_idx": ref_idx,
        "scores": scores,

        "rpeaks_ref": rpeaks_ref,
        "rpeaks_by_lead": rpeaks_by_lead,

        "tpeaks_by_lead": tpeaks_by_lead,
        "ton_by_lead": ton_by_lead,
        "toff_by_lead": toff_by_lead,

        "TpMat": TpMat,
        "TonMat": TonMat,
        "ToffMat": ToffMat,

        "TonToff_pair_idx": TonToff_pair,        # (12, beats, 2)
        "TonToff_pair_sec": TonToff_pair_sec,    # (12, beats, 2)

        "valid_mask": valid_mask,

        "consensus": {
            "tonset_sec": ton_cons_sec,
            "tpeak_sec": tp_cons_sec,
            "toffset_sec": toff_cons_sec,
        },
    }
