import numpy as np
from Phase1.Preprocessing import FilterTransform
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_out_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_split(split: str = "test") -> np.ndarray:
    x_path = repo_root() / "dataset" / f"X_{split}.npy"
    return np.load(x_path)


class TwavePreprocessor:
    """
    Preprocessing for T-wave delineation:
    - baseline correction
    - low-pass filtering
    - optional z-score normalization for R-peak detection
    """

    def __init__(self, fs=100, k1=21, k2=61, lp_cut=35, lp_taps=101):
        # Filtered signal for T-wave analysis
        self.tf_no_std = FilterTransform(
            fs=fs, k1=k1, k2=k2, lp_cut=lp_cut, lp_taps=lp_taps, standardize=False
        )
        # Standardized signal for R-peak detection
        self.tf_std = FilterTransform(
            fs=fs, k1=k1, k2=k2, lp_cut=lp_cut, lp_taps=lp_taps, standardize=True
        )

    def __call__(
        self,
        ecg: np.ndarray,
        return_standardized_for_rpeak: bool = True,
        out_layout: str = "12xL",
    ):
        """
        Input:
            ecg: shape (L, 12) or (12, L)
            out_layout: "12xL" or "Lx12"

        Return:
            ecg_filt: filtered ECG without z-score normalization
            ecg_std: standardized ECG (optional)
        """
        if ecg.ndim != 2:
            raise ValueError(f"ecg must be 2D, got shape={ecg.shape}")

        ecg_filt = self.tf_no_std(ecg).numpy().astype(np.float32, copy=False)
        if return_standardized_for_rpeak:
            ecg_std = self.tf_std(ecg).numpy().astype(np.float32, copy=False)
        else:
            ecg_std = None

        def to_layout(x):
            if x is None:
                return None
            if out_layout == "12xL":
                return x if x.shape[0] == 12 else x.T
            elif out_layout == "Lx12":
                return x if x.shape[1] == 12 else x.T
            else:
                raise ValueError("out_layout must be '12xL' or 'Lx12'")

        return to_layout(ecg_filt), to_layout(ecg_std)
