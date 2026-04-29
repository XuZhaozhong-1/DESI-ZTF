#!/usr/bin/env python3
"""
Fit sigma_m(m) separately for each (bin, case) where case ∈ {all, upper, lower}.

Fit method:
  - select points by case
  - bin in m
  - compute median y_j = median(log10(sigma_m)) per bin
  - fit polynomial in x=(m-m_ref) to y_j with weights w_j = sqrt(n_j)

Reads:
  data/processed/y3_bin{idx}_z{zmin}-{zmax}_arrays.npz
  data/processed/stripes/bin{idx}_z{zmin}-{zmax}_stripes.npz   (upper/lower masks)

Writes:
  results/sigma_m_models/
    sigma_m_model_bin{idx}_z{zmin}-{zmax}_{case}.npz
    sigma_m_fit_bin{idx}_z{zmin}-{zmax}_{case}.png
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.home() / "project"
DATA = ROOT / "data" / "processed"
STRP = DATA / "stripes"
OUT  = ROOT / "results" / "sigma_m_models"
OUT.mkdir(parents=True, exist_ok=True)

BINS = {
    1: (2.3, 2.4),
    2: (2.4, 2.5),
    3: (2.5, 2.6),
    4: (2.6, 2.7),
    5: (2.7, 2.8),
}
CASES = ("all", "upper", "lower")

def load_arrays(idx, zmin, zmax):
    npz = DATA / f"y3_bin{idx}_z{zmin}-{zmax}_arrays.npz"
    d = np.load(npz)
    m = d["apr_mag_r"].astype(np.float64)
    s = d["sigma_m"].astype(np.float64)
    return m, s, npz

def load_stripes(idx, zmin, zmax):
    tag = f"bin{idx}_z{zmin}-{zmax}"
    sp = STRP / f"{tag}_stripes.npz"
    s = np.load(sp)
    upper = s["upper"].astype(bool)
    lower = s["lower"].astype(bool)
    return upper, lower, sp

def fit_binned_median_poly(m, sigma_m, mask, m_ref=22.0, nbins=20, min_per_bin=20, deg=1):
    """
    Fit y=log10(sigma_m) vs m using binned medians and weights sqrt(n_j).
    """
    m = np.asarray(m)
    sigma_m = np.asarray(sigma_m)
    mask = np.asarray(mask, dtype=bool)

    if mask.sum() < 200:
        raise RuntimeError(f"Too few points in this case: N={int(mask.sum())}")

    m_u = m[mask]
    y_u = np.log10(sigma_m[mask])

    edges = np.linspace(m_u.min(), m_u.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    m_centers, y_med, n_cnt = [], [], []
    for i in range(nbins):
        sel = (m_u >= edges[i]) & (m_u < edges[i+1])
        n = int(sel.sum())
        if n < min_per_bin:
            continue
        m_centers.append(centers[i])
        y_med.append(np.median(y_u[sel]))
        n_cnt.append(n)

    m_centers = np.asarray(m_centers, float)
    y_med     = np.asarray(y_med, float)
    n_cnt     = np.asarray(n_cnt, int)

    if len(m_centers) < (deg + 2):
        raise RuntimeError(f"Not enough populated bins: nbins_used={len(m_centers)} (need >= {deg+2}).")

    x = m_centers - m_ref
    w = np.sqrt(n_cnt)  # your requested "gentler" weighting
    coeff = np.polyfit(x, y_med, deg=deg, w=w)  # highest power first

    info = dict(
        deg=int(deg),
        m_ref=float(m_ref),
        nbins=int(nbins),
        min_per_bin=int(min_per_bin),
        nbins_used=int(len(m_centers)),
        m_centers=m_centers,
        y_median=y_med,
        n_per_bin=n_cnt,
        weights=w,
        N_case=int(mask.sum()),
        N_fit=int(n_cnt.sum()),
        m_min=float(m_u.min()),
        m_max=float(m_u.max()),
    )
    return coeff, info

def plot_fit(m, sigma_m, mask, coeff, info, outpng, title=""):
    m = np.asarray(m)
    y = np.log10(np.asarray(sigma_m))

    xx = np.linspace(m.min(), m.max(), 400)
    yy = np.polyval(coeff, xx - info["m_ref"])

    plt.figure(figsize=(6.8, 5.0))
    plt.scatter(m, y, s=2, alpha=0.10, label="all")
    plt.scatter(m[mask], y[mask], s=3, alpha=0.25, label="case")
    plt.scatter(info["m_centers"], info["y_median"], s=40, c="k", zorder=5, label="binned median")
    plt.plot(xx, yy, "k", lw=2, label=f"deg={info['deg']} fit")
    plt.xlabel("m")
    plt.ylabel(r"$\log_{10}\sigma_m$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=170)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", default="1,2,3,4,5")
    ap.add_argument("--cases", default="all,upper,lower")
    ap.add_argument("--m-ref", type=float, default=22.0)
    ap.add_argument("--deg", type=int, default=1)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--min-per-bin", type=int, default=20)
    args = ap.parse_args()

    bin_list = [int(x) for x in args.bins.split(",") if x.strip()]
    case_list = [x.strip() for x in args.cases.split(",") if x.strip()]
    for c in case_list:
        if c not in CASES:
            raise SystemExit(f"Unknown case: {c}")

    for idx in bin_list:
        zmin, zmax = BINS[idx]
        tag = f"bin{idx}_z{zmin}-{zmax}"

        m, sigma_m, arr_path = load_arrays(idx, zmin, zmax)
        upper, lower, stripe_path = load_stripes(idx, zmin, zmax)

        for case in case_list:
            if case == "all":
                mask = np.ones_like(m, dtype=bool)
            elif case == "upper":
                mask = upper
            elif case == "lower":
                mask = lower
            else:
                raise RuntimeError("unreachable")

            coeff, info = fit_binned_median_poly(
                m, sigma_m, mask,
                m_ref=args.m_ref, nbins=args.nbins, min_per_bin=args.min_per_bin, deg=args.deg
            )

            out_npz = OUT / f"sigma_m_model_{tag}_{case}.npz"
            np.savez(
                out_npz,
                coeff=coeff,
                **info,
                zmin=float(zmin),
                zmax=float(zmax),
                N_all=int(len(m)),
                arrays_file=str(arr_path),
                stripes_file=str(stripe_path),
                case=case,
            )

            out_png = OUT / f"sigma_m_fit_{tag}_{case}.png"
            plot_fit(m, sigma_m, mask, coeff, info, out_png,
                     title=f"{tag} {case}: N_case={info['N_case']:,}, N_fit={info['N_fit']:,}")

            print(f"[{tag} {case}] wrote {out_npz.name} and {out_png.name}")

if __name__ == "__main__":
    main()
