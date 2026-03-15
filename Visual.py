# SpecialTask/Visual.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_rpeak_comparison_leads(
    res: dict,
    fs: int = 100,
    lead_indices: Tuple[int, int, int] = (2, 3, 7),
    xlim: Tuple[float, float] = (0.0, 10.0),
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 200,
    title: str = "R-peak Detection on Selected Leads",
) -> plt.Figure:
    """
    Plot R-peaks on selected leads (default: Lead 02, 03, 07).
    """
    ecg_filt = np.asarray(res["ecg_filt"], dtype=float)
    rpeaks_by_lead = np.asarray(res["rpeaks_by_lead"], dtype=int)

    if ecg_filt.ndim != 2 or ecg_filt.shape[0] != 12:
        raise ValueError(f"ecg_filt must be (12, L), got {ecg_filt.shape}")

    if rpeaks_by_lead.ndim != 2 or rpeaks_by_lead.shape[0] != 12:
        raise ValueError(f"rpeaks_by_lead must be (12, n_beats), got {rpeaks_by_lead.shape}")

    leads_to_plot = list(lead_indices)
    for li in leads_to_plot:
        if not (0 <= li < 12):
            raise ValueError(f"lead index must be in [0, 11], got {li}")

    L = ecg_filt.shape[1]
    t = np.arange(L) / fs

    fig, axes = plt.subplots(len(leads_to_plot), 1, figsize=(14, 8), sharex=True)
    if len(leads_to_plot) == 1:
        axes = [axes]

    for ax, li in zip(axes, leads_to_plot):
        sig = ecg_filt[li]
        rp = rpeaks_by_lead[li]
        rp = rp[(rp >= 0) & (rp < L)]

        ax.plot(t, sig, linewidth=1.0)
        if len(rp) > 0:
            ax.scatter(t[rp], sig[rp], s=40, c="tab:blue", zorder=3)

        ax.set_xlim(*xlim)
        ax.grid(True)
        ax.text(
            0.01, 0.90, f"Lead {li:02d}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )
        ax.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time (s)")

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="R-peaks",
               markerfacecolor="tab:blue", markeredgecolor="tab:blue", markersize=6),
    ]

    fig.legend(
        handles=legend_elems,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        fontsize=11,
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        print(f"[OK] Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_twave_detection_12lead(
    res: dict,
    fs: int = 100,
    out_path: Optional[Union[str, Path]] = None,
    xlim: Tuple[float, float] = (0.0, 10.0),
    show: bool = False,
    dpi: int = 200,
    title: str = "T-wave detection (per lead) with R-peaks, onset, peak, offset",
    leadsrange: range = range(0, 12),
    shade_twave: bool = True,
) -> plt.Figure:
    """
    Plot 12-lead T-wave detection results with R-peaks, T-peaks,
    onset/offset markers, and optional T-wave shading.
    """
    ecg_filt = np.asarray(res["ecg_filt"])
    tpeaks_by_lead = res["tpeaks_by_lead"]
    ton_by_lead = res["ton_by_lead"]
    toff_by_lead = res["toff_by_lead"]
    rpeaks_by_lead = np.asarray(res["rpeaks_by_lead"], dtype=int)

    if ecg_filt.ndim != 2 or ecg_filt.shape[0] != 12:
        raise ValueError(f"ecg_filt must be (12, L), got {ecg_filt.shape}")

    if rpeaks_by_lead.ndim != 2 or rpeaks_by_lead.shape[0] != 12:
        raise ValueError(f"rpeaks_by_lead must be (12, n_beats), got {rpeaks_by_lead.shape}")

    L = ecg_filt.shape[1]
    t = np.arange(L) / fs
    leads_to_plot = list(leadsrange)

    fig, axes = plt.subplots(len(leads_to_plot), 1, figsize=(16, 20), sharex=True)
    if len(leads_to_plot) == 1:
        axes = [axes]

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="R-peaks",
               markerfacecolor="tab:blue", markeredgecolor="tab:blue", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="T-peak",
               markerfacecolor="tab:orange", markeredgecolor="tab:orange", markersize=6),
        Line2D([0], [0], marker="x", color="tab:green", label="T-onset",
               markersize=7, linestyle="None"),
        Line2D([0], [0], marker="x", color="tab:red", label="T-offset",
               markersize=7, linestyle="None"),
        Line2D([0], [0], color="tab:blue", linewidth=6, alpha=0.10, label="T-wave interval"),
    ]

    for ax, li in zip(axes, leads_to_plot):
        sig = ecg_filt[li]

        rp = np.asarray(rpeaks_by_lead[li], dtype=int)
        rp = rp[(rp >= 0) & (rp < L)]

        tp = np.asarray(tpeaks_by_lead.get(li, []), dtype=int)
        ton = np.asarray(ton_by_lead.get(li, []), dtype=int)
        toff = np.asarray(toff_by_lead.get(li, []), dtype=int)

        n_beats = max(len(tp), len(ton), len(toff))
        if len(tp) < n_beats:
            tp = np.pad(tp, (0, n_beats - len(tp)), constant_values=-1)
        if len(ton) < n_beats:
            ton = np.pad(ton, (0, n_beats - len(ton)), constant_values=-1)
        if len(toff) < n_beats:
            toff = np.pad(toff, (0, n_beats - len(toff)), constant_values=-1)

        valid_tp = (tp >= 0) & (tp < L)
        valid_ton = (ton >= 0) & (ton < L)
        valid_toff = (toff >= 0) & (toff < L)

        ax.plot(t, sig, linewidth=1.0)
        ax.set_xlim(*xlim)

        if rp.size > 0:
            ax.scatter(t[rp], sig[rp], s=40, c="tab:blue", zorder=3)

        if valid_tp.any():
            ax.scatter(t[tp[valid_tp]], sig[tp[valid_tp]], s=40, c="tab:orange", zorder=4)

        if valid_ton.any():
            ax.scatter(
                t[ton[valid_ton]],
                sig[ton[valid_ton]],
                s=40,
                marker="x",
                c="tab:green",
                zorder=4,
            )

        if valid_toff.any():
            ax.scatter(
                t[toff[valid_toff]],
                sig[toff[valid_toff]],
                s=40,
                marker="x",
                c="tab:red",
                zorder=4,
            )

        if shade_twave:
            for b in range(n_beats):
                ton_b = ton[b]
                toff_b = toff[b]

                if ton_b < 0 or toff_b < 0:
                    continue
                if ton_b >= L or toff_b >= L:
                    continue
                if toff_b <= ton_b:
                    continue

                ax.axvspan(
                    ton_b / fs,
                    toff_b / fs,
                    color="tab:blue",
                    alpha=0.08,
                    zorder=0
                )

        ax.grid(True)
        ax.text(
            0.01, 0.92, f"Lead {li} (filtered)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )

    fig.legend(
        handles=legend_elems,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=5,
        frameon=True,
        fontsize=12,
        markerscale=1.3,
        handletextpad=0.6,
        columnspacing=1.2,
        borderpad=0.4,
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        print(f"[OK] Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_processing_display(
    res: dict,
    fs: int = 100,
    lead_idx: int = 9,
    xlim: Tuple[float, float] = (2.0, 8.0),
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 200,
    title: Optional[str] = None,
    shade_twave: bool = True,
) -> plt.Figure:
    """
    Plot a step-by-step processing display for one lead:
    preprocessing, beat detection, T-peak detection, and Ton/Toff detection.
    """
    ecg_filt = np.asarray(res["ecg_filt"], dtype=float)
    rpeaks_by_lead = np.asarray(res["rpeaks_by_lead"], dtype=int)
    tpeaks_by_lead = res["tpeaks_by_lead"]
    ton_by_lead = res["ton_by_lead"]
    toff_by_lead = res["toff_by_lead"]

    if ecg_filt.ndim != 2 or ecg_filt.shape[0] != 12:
        raise ValueError(f"ecg_filt must be (12, L), got {ecg_filt.shape}")

    if rpeaks_by_lead.ndim != 2 or rpeaks_by_lead.shape[0] != 12:
        raise ValueError(f"rpeaks_by_lead must be (12, n_beats), got {rpeaks_by_lead.shape}")

    if not (0 <= lead_idx < 12):
        raise ValueError(f"lead_idx must be in [0, 11], got {lead_idx}")

    sig = ecg_filt[lead_idx]
    L = len(sig)
    t = np.arange(L) / fs

    rp = np.asarray(rpeaks_by_lead[lead_idx], dtype=int)
    tp = np.asarray(tpeaks_by_lead.get(lead_idx, []), dtype=int)
    ton = np.asarray(ton_by_lead.get(lead_idx, []), dtype=int)
    toff = np.asarray(toff_by_lead.get(lead_idx, []), dtype=int)

    rp = rp[(rp >= 0) & (rp < L)]
    tp = tp[(tp >= 0) & (tp < L)]
    ton = ton[(ton >= 0) & (ton < L)]
    toff = toff[(toff >= 0) & (toff < L)]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    if title is None:
        title = f"Processing Display - Lead {lead_idx:02d}"

    # 1. preprocessing
    axes[0].plot(t, sig, linewidth=1.0)
    axes[0].set_title("1. Preprocessing")
    axes[0].grid(True)
    axes[0].set_xlim(*xlim)

    # 2. beat detection
    axes[1].plot(t, sig, linewidth=1.0)
    if len(rp) > 0:
        axes[1].scatter(t[rp], sig[rp], s=35, c="tab:blue", label="R-peaks", zorder=3)
    axes[1].set_title("2. Beat Detection")
    axes[1].grid(True)
    axes[1].set_xlim(*xlim)

    # 3. T-peak detection
    axes[2].plot(t, sig, linewidth=1.0)
    if len(rp) > 0:
        axes[2].scatter(t[rp], sig[rp], s=30, c="tab:blue", label="R-peaks", zorder=3)
    if len(tp) > 0:
        axes[2].scatter(t[tp], sig[tp], s=35, c="tab:orange", label="T-peaks", zorder=4)
    axes[2].set_title("3. T-peak Detection")
    axes[2].grid(True)
    axes[2].set_xlim(*xlim)

    # 4. Tonset / Toffset detection
    axes[3].plot(t, sig, linewidth=1.0)

    if len(rp) > 0:
        axes[3].scatter(t[rp], sig[rp], s=28, c="tab:blue", label="R-peaks", zorder=3)
    if len(tp) > 0:
        axes[3].scatter(t[tp], sig[tp], s=35, c="tab:orange", label="T-peaks", zorder=4)
    if len(ton) > 0:
        axes[3].scatter(t[ton], sig[ton], s=40, marker="x", c="tab:green", label="Tonset", zorder=5)
    if len(toff) > 0:
        axes[3].scatter(t[toff], sig[toff], s=40, marker="x", c="tab:red", label="Toffset", zorder=5)

    if shade_twave:
        n_pairs = min(len(ton), len(toff))
        for i in range(n_pairs):
            if toff[i] > ton[i]:
                axes[3].axvspan(ton[i] / fs, toff[i] / fs, color="tab:blue", alpha=0.08, zorder=0)

    axes[3].set_title("4. Tonset / Toffset Detection")
    axes[3].grid(True)
    axes[3].set_xlim(*xlim)

    for ax in axes:
        ax.set_ylabel("Amplitude")
    axes[-1].set_xlabel("Time (s)")

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="R-peaks",
               markerfacecolor="tab:blue", markeredgecolor="tab:blue", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="T-peaks",
               markerfacecolor="tab:orange", markeredgecolor="tab:orange", markersize=6),
        Line2D([0], [0], marker="x", color="tab:green", label="Tonset",
               linestyle="None", markersize=7),
        Line2D([0], [0], marker="x", color="tab:red", label="Toffset",
               linestyle="None", markersize=7),
        Line2D([0], [0], color="tab:blue", linewidth=6, alpha=0.10, label="T-wave interval"),
    ]

    fig.legend(
        handles=legend_elems,
        loc="upper right",
        ncol=5,
        bbox_to_anchor=(0.985, 0.98),
        frameon=True,
        fontsize=11,
    )

    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        print(f"[OK] Saved figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_all(
    res: dict,
    fs: int = 100,
    out_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 200,
) -> dict:
    """
    Plot and optionally save:
    1) R-peak comparison
    2) 12-lead T-wave detection
    3) Processing display
    """
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rpeak_compare_path = out_dir / "rpeak_comparison.png"
        twave_12lead_path = out_dir / "tpeaks_leads.png"
        processing_display_path = out_dir / "process_display.png"
    else:
        rpeak_compare_path = None
        twave_12lead_path = None
        processing_display_path = None

    plot_rpeak_comparison_leads(
        res=res,
        fs=fs,
        lead_indices=(2, 3, 7),
        xlim=(0.0, 10.0),
        out_path=rpeak_compare_path,
        show=show,
        dpi=dpi,
        title="R-peak Detection on Selected Leads",
    )

    plot_twave_detection_12lead(
        res=res,
        fs=fs,
        out_path=twave_12lead_path,
        xlim=(0.0, 10.0),
        show=show,
        dpi=dpi,
        title="T-wave detection (per lead) with R-peaks, onset, peak, offset",
    )

    plot_processing_display(
        res=res,
        fs=fs,
        lead_idx=9,
        xlim=(2.0, 8.0),
        out_path=processing_display_path,
        show=show,
        dpi=dpi,
        title="Processing Display - Lead 09",
    )

    return {
        "rpeak_compare": rpeak_compare_path,
        "twave_12lead": twave_12lead_path,
        "processing_display": processing_display_path,
    }


def print_rpeaks_matrix_with_polarity_and_score(
    res: dict,
    fs: int = 100,
    ndigits: int = 2,
) -> None:
    """
    Print lead-wise R-peaks (seconds), polarity, and score.
    """
    rpeaks_ref = np.asarray(res["rpeaks_ref"], dtype=int)
    ecg_filt = np.asarray(res["ecg_filt"], dtype=float)
    rpeaks_by_lead = np.asarray(res["rpeaks_by_lead"], dtype=int)
    scores = np.asarray(res["scores"], dtype=float)
    ref_idx = int(res["ref_idx"])

    if ecg_filt.ndim != 2 or ecg_filt.shape[0] != 12:
        raise ValueError(f"ecg_filt must be (12, L), got {ecg_filt.shape}")

    if rpeaks_by_lead.ndim != 2 or rpeaks_by_lead.shape[0] != 12:
        raise ValueError(f"rpeaks_by_lead must be (12, n_beats), got {rpeaks_by_lead.shape}")

    print("\n--- RPEAK MATRIX (sec) + POLARITY + SCORE ---")

    n_leads, n_beats = rpeaks_by_lead.shape
    L = ecg_filt.shape[1]

    for li in range(n_leads):
        rp = rpeaks_by_lead[li]
        valid = (rp >= 0) & (rp < L)

        rp_sec = np.full(n_beats, np.nan, dtype=float)
        rp_sec[valid] = rp[valid] / fs

        if np.any(valid):
            vals = ecg_filt[li, rp[valid]]
            polarity = "pos" if np.median(vals) >= 0 else "neg"
        else:
            polarity = "unknown"

        arr_str = np.array2string(
            rp_sec,
            precision=ndigits,
            separator=" ",
            suppress_small=False,
            floatmode="fixed",
        )

        print(f"lead {li:02d}: {arr_str}  {polarity}  score={scores[li]:.2f}")

    print(f"\nreference lead = lead {ref_idx:02d}")
    print(f"num-rpeaks = {len(rpeaks_ref)}")


def build_tpeak_matrix(
    tpeaks_by_lead: dict,
    n_beats: int,
    n_leads: int = 12,
    pad_value: int = -1,
    verbose: bool = True,
):
    """
    Align lead-wise arrays into an (n_beats, n_leads) matrix.
    Short arrays are padded; long arrays are truncated.
    """
    Tmat = np.full((n_beats, n_leads), pad_value, dtype=int)

    for li in range(n_leads):
        tp = np.asarray(tpeaks_by_lead[li], dtype=int)

        if tp.shape[0] == n_beats:
            Tmat[:, li] = tp
        else:
            m = min(n_beats, tp.shape[0])
            Tmat[:m, li] = tp[:m]
            if verbose:
                print(f"[WARN] lead {li}: tpeaks length={tp.shape[0]} != n_beats={n_beats}, padded with {pad_value}")

    return Tmat


def print_tpeak_matrix_by_lead(TpMat: np.ndarray, fs: int = 100, ndigits: int = 2) -> None:
    """
    Print the T-peak matrix row by row by lead.
    """
    TpMat = np.asarray(TpMat, dtype=int)
    if TpMat.ndim != 2 or TpMat.shape[1] != 12:
        raise ValueError(f"TpMat must be (n_beats, 12), got {TpMat.shape}")

    TpSec = np.where(TpMat >= 0, TpMat / fs, np.nan)
    TpSec_by_lead = TpSec.T

    print("\n--- TPEAK MATRIX (sec) ---")
    for li in range(TpSec_by_lead.shape[0]):
        arr_str = np.array2string(
            TpSec_by_lead[li],
            precision=ndigits,
            separator=" ",
            suppress_small=False,
            floatmode="fixed",
        )
        print(f"lead {li:02d}: {arr_str}")


def print_ton_toff_pair_matrix(
    pair_sec: np.ndarray,
    valid_mask: np.ndarray,
    ndigits: int = 2,
) -> None:
    """
    Print lead-wise Ton/Toff pairs with valid counts.
    """
    pair_sec = np.asarray(pair_sec, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if pair_sec.ndim != 3 or pair_sec.shape[0] != 12 or pair_sec.shape[2] != 2:
        raise ValueError(f"expect pair_sec shape (12, n_beats, 2), got {pair_sec.shape}")

    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must be 2D, got shape={valid_mask.shape}")

    Ton = pair_sec[:, :, 0]
    Toff = pair_sec[:, :, 1]

    n_leads, n_beats = Ton.shape

    if valid_mask.shape != (n_beats, n_leads):
        raise ValueError(
            f"expect valid_mask shape {(n_beats, n_leads)}, got {valid_mask.shape}"
        )

    counts = np.sum(valid_mask, axis=0)

    print("\n=== [Tonset, Toffset] pair matrix (12 x beats) ===")
    for i in range(n_leads):
        cells = []
        for j in range(n_beats):
            a, b = Ton[i, j], Toff[i, j]
            if np.isnan(a) or np.isnan(b):
                cells.append("[nan,nan]")
            else:
                cells.append(f"[{a:.{ndigits}f},{b:.{ndigits}f}]")

        row = " ".join(cells)
        print(f"lead {i:02d}: {row}  {int(counts[i])}/{n_beats}")


def print_summary(res: dict, fs: int = 100):
    """
    Print a summary of the pipeline results.
    """
    print("\n========== PIPELINE SUMMARY ==========")

    print_rpeaks_matrix_with_polarity_and_score(res, fs=fs, ndigits=2)

    print_tpeak_matrix_by_lead(res["TpMat"], fs=fs, ndigits=2)

    print("\n--- [Tonset, Toffset] ---")
    print("Valid T-wave criterion: Tonset < Tpeak < Toffset and duration >= 40 ms")
    pair = res["TonToff_pair_sec"]
    valid_mask = res["valid_mask"]
    print_ton_toff_pair_matrix(pair, valid_mask, ndigits=2)

    c = res["consensus"]

    print("\n--- CONSENSUS ---")
    print("Tonset(sec):", np.round(c["tonset_sec"], 3).tolist())
    print("Tpeak(sec):", np.round(c["tpeak_sec"], 3).tolist())
    print("Toffset(sec):", np.round(c["toffset_sec"], 3).tolist())