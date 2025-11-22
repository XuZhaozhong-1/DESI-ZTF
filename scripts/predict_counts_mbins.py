#!/usr/bin/env python3
"""
Predict DESI quasar counts per magnitude bin using your fitted QLF+efficiency model.

For each fitted redshift bin:
  1) Split the DESI magnitudes in that bin into 20 equal-width bins (configurable).
  2) Evaluate your intensity I(m) with posterior draws.
  3) Integrate I(m) over each m-bin to get predicted counts.
  4) Save observed vs predicted (mean, 16–84%) to CSV and a quick overlay plot.

Defaults assume:
  - arrays:   ~/project/data/processed/y3_bin{idx}_z{zmin}-{zmax}_arrays.npz
  - posteriors: ~/project/results/mcmc/per_bin/y3_bin{idx}_z{zmin}-{zmax}.npz
"""

import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

# -------- helpers --------
def softplus_stable(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-ax))

def load_bin_arrays(arr_dir: Path, idx: int, zmin: float, zmax: float):
    f = arr_dir / f"y3_bin{idx}_z{zmin}-{zmax}_arrays.npz"
    d = np.load(f)
    m       = d["apr_mag_r"].astype(float)
    sigma_m = d["sigma_m"].astype(float)
    k       = d["k"].astype(float)
    sigma_k = d["sigma_k"].astype(float)
    mu      = d["mu"].astype(float)
    return m, sigma_m, k, sigma_k, mu, f

def load_posterior(post_dir: Path, idx: int, zmin: float, zmax: float):
    f = post_dir / f"y3_bin{idx}_z{zmin}-{zmax}.npz"
    p = np.load(f)
    # accept either nat-log or base-10 keys
    ln10 = np.log(10.0)
    if "log_Nphi" in p:
        log_Nphi  = p["log_Nphi"]
    else:
        log_Nphi  = p["log10_Nphi"] * ln10
    if "log_Lstar" in p:
        log_Lstar = p["log_Lstar"]
    else:
        log_Lstar = p["log10_Lstar"] * ln10
    out = {
        "m0": p["m0"], "b": p["b"], "alpha": p["alpha"], "beta": p["beta"],
        "log_Nphi": log_Nphi, "log_Lstar": log_Lstar
    }
    return out, f

def intensity_log(m, mu, sigma_m, k, sigma_k, m0, b, alpha, beta, log_Nphi, log_Lstar,
                  ln10, ln_Lbol0):
    # weighted mean k
    w = 1.0 / (sigma_k**2)
    w[~np.isfinite(w)] = 0.0
    k_mean = (k * w).sum() / max(w.sum(), 1e-300)

    sigm = float(sigma_m.mean())
    M0   = m - mu - (ln10/2.5) * (alpha * sigm**2) - k_mean

    lnL0 = ln_Lbol0 - 0.4 * ln10 * M0 - log_Lstar
    log_den = np.logaddexp( -(alpha+1.0)*lnL0, -(beta+1.0)*lnL0 )

    log_eff = -softplus_stable(b * (m - m0))
    exponent = - ((ln10/2.5) * alpha * sigm**2)**2 / (2.0 * sigm**2)

    # final log I
    return log_Nphi + log_eff + exponent - log_den + np.log(ln10/2.5)

def predict_counts_per_mbin(m, mu, sigma_m, k, sigma_k, params, edges, ln10, ln_Lbol0):
    """
    For one posterior draw (params dict), return predicted counts per m-bin
    by integrating I(m) over the subset of sample m in that bin (trapz on sorted m).
    """
    logI = intensity_log(m, mu, sigma_m, k, sigma_k,
                         params["m0"], params["b"], params["alpha"], params["beta"],
                         params["log_Nphi"], params["log_Lstar"],
                         ln10, ln_Lbol0)
    # sort once
    order = np.argsort(m)
    m_sorted   = m[order]
    I_sorted   = np.exp(logI[order])

    preds = np.zeros(len(edges)-1, dtype=float)
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        sel = (m_sorted >= lo) & (m_sorted < hi if i < len(edges)-2 else m_sorted <= hi)
        if sel.sum() >= 2:
            preds[i] = np.trapz(I_sorted[sel], m_sorted[sel])
        else:
            preds[i] = 0.0
    return preds

def parse_bins(s: str):
    """
    Parse a string like: "1:2.3-2.4,2:2.4-2.5,3:2.5-2.6,4:2.6-2.8"
    into a list of (idx, zmin, zmax).
    """
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        i_part, z_part = tok.split(":")
        zmin_s, zmax_s = z_part.split("-")
        out.append((int(i_part), float(zmin_s), float(zmax_s)))
    return out

def main():
    home = Path.home()
    default_root = home / "project"

    ap = argparse.ArgumentParser(description="Predict DESI counts per m-bin from fitted QLF.")
    ap.add_argument("--arr-dir", default=str(default_root / "data/processed"))
    ap.add_argument("--post-dir", default=str(default_root / "results/mcmc/per_bin"))
    ap.add_argument("--out-table-dir", default=str(default_root / "results/tables"))
    ap.add_argument("--out-fig-dir", default=str(default_root / "results/figures/pred_counts"))
    ap.add_argument("--bins", default="1:2.3-2.4,2:2.4-2.5,3:2.5-2.6,4:2.6-2.8",
                    help='Format: "1:2.3-2.4,2:2.4-2.5,3:2.5-2.6,4:2.6-2.8"')
    ap.add_argument("--m-bins", type=int, default=20)
    ap.add_argument("--draws", type=int, default=400, help="Posterior draws per bin for prediction.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    arr_dir     = Path(args.arr_dir)
    post_dir    = Path(args.post_dir)
    out_tab_dir = Path(args.out_table_dir); out_tab_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = Path(args.out_fig_dir);   out_fig_dir.mkdir(parents=True, exist_ok=True)

    bins = parse_bins(args.bins)
    rng = np.random.default_rng(args.seed)
    ln10 = np.log(10.0)
    ln_Lbol0 = np.log(const.L_bol0.to(u.erg/u.s).value)

    for idx, zmin, zmax in bins:
        # ---- load data and posterior ----
        m, sigma_m, k, sigma_k, mu, arr_file = load_bin_arrays(arr_dir, idx, zmin, zmax)
        post, post_file = load_posterior(post_dir, idx, zmin, zmax)
        n_post = len(post["m0"])
        draws  = rng.choice(n_post, size=min(args.draws, n_post), replace=False)

        # ---- magnitude binning (equal-width over DESI m in this z-bin) ----
        m_lo, m_hi = float(m.min()), float(m.max())
        edges = np.linspace(m_lo, m_hi, args.m_bins+1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        obs_counts, _ = np.histogram(m, bins=edges)

        # ---- predicted counts across posterior draws ----
        pred_mat = np.zeros((len(draws), args.m_bins), dtype=float)
        for j, s in enumerate(draws):
            params = {k: post[k][s] for k in post}
            pred_mat[j] = predict_counts_per_mbin(
                m, mu, sigma_m, k, sigma_k, params, edges, ln10, ln_Lbol0
            )

        pred_mean = np.nanmean(pred_mat, axis=0)
        pred_lo, pred_hi = np.nanpercentile(pred_mat, [16,84], axis=0)

        # ---- save CSV ----
        out_csv = out_tab_dir / f"pred_counts_m{args.m_bins}_bin{idx}_z{zmin}-{zmax}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["zmin","zmax","m_lo","m_hi","m_center","N_obs","N_pred_mean","N_pred_p16","N_pred_p84"])
            for i in range(args.m_bins):
                w.writerow([zmin, zmax, edges[i], edges[i+1], centers[i],
                            int(obs_counts[i]), pred_mean[i], pred_lo[i], pred_hi[i]])
        print(f"[bin {idx}] wrote {out_csv}")

        # ---- overlay plot ----
        fig, ax = plt.subplots(figsize=(6.4,4.2))
        width = (edges[1]-edges[0]) * 0.9
        ax.bar(centers, obs_counts, width=width, alpha=0.45, label="DESI observed")
        ax.plot(centers, pred_mean, lw=2, label="Predicted (mean)")
        ax.fill_between(centers, pred_lo, pred_hi, alpha=0.20, label="Predicted (16–84%)")
        ax.set_xlabel(r"$m_R$")
        ax.set_ylabel("Counts per mag-bin")
        ax.set_title(f"z={zmin}-{zmax} (m bins={args.m_bins})")
        ax.legend()
        fig.tight_layout()
        out_fig = out_fig_dir / f"pred_counts_m{args.m_bins}_bin{idx}_z{zmin}-{zmax}.png"
        fig.savefig(out_fig, dpi=200)
        plt.close(fig)
        print(f"[bin {idx}] wrote {out_fig}")

    print("Done.")

if __name__ == "__main__":
    main()
