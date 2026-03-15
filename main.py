from pathlib import Path
import numpy as np

from .Pipeline import detect_twaves_12lead
from .Visual import print_summary, plot_all
from .Preprocess import load_split, get_out_dir

def main():

    # ---------- init ----------
    fs = 100
    X = load_split("test")
    x = X[0]
    out_dir = get_out_dir()

    # ---------- detect ----------
    res = detect_twaves_12lead(x, fs=fs)

    # ---------- summary ----------
    print_summary(res, fs)

    # ---------- plot ----------
    plot_all(
        res=res,
        fs=fs,
        out_dir=out_dir,
        show=False,
        dpi=200,
    )

if __name__ == "__main__":
    main()