#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from y3pipe.desi import read_filter

ROOT = Path.home() / "project"
R_FILTER_LOCAL = ROOT / "data/raw/filters/decam_r.dat"
OUTDIR = ROOT / "results/figures/filters"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load DECam r filter curve
    lam_r, t_r = read_filter(R_FILTER_LOCAL)   # wavelength [Å], throughput

    plt.figure(figsize=(6,4))
    plt.plot(lam_r, t_r, lw=1.8)
    plt.xlabel(r"Wavelength [$\mathrm{\AA}$]")
    plt.ylabel("Throughput")
    plt.title("DECam $r$-band filter response")
    plt.xlim(lam_r.min(), lam_r.max())
    plt.ylim(0, 1.05 * np.nanmax(t_r))
    plt.tight_layout()

    out = OUTDIR / "decam_r_curve.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
