#!/usr/bin/env python3
"""
Predict DESI quasar counts per magnitude bin using the fitted QLF+efficiency model.

For each fitted redshift bin:
  1) Split the DESI magnitudes in that bin into equal-width bins (configurable).
  2) Evaluate the intensity I(m) with posterior draws.
  3) Integrate I(m) over each m-bin to get predicted counts.
  4) Save observed vs predicted (mean, 16–84%) to CSV and an overlay plot.

Defaults assume:
  - arrays:
      ~/project/data/processed/y3_bin{idx}_z{zmin}-{zmax}_arrays.npz
  - posteriors:
      ~/project/results/mcmc/per_bin/order1_all/
      y3_bin{idx}_z{zmin}-{zmax}_all_order1_w10000_s10000_c4.npz
"""

import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

# -------- helpers --------
def softplus_stable(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-ax))

def load_bin_arrays(arr_dir: Path, idx: int, zmin: float, zmax: float):
    f = arr_dir / f"y3_bin{idx}_z{zmin}-{zmax}_arrays.npz"
    d = np.load(f)
    m = d["apr_mag_r"].astype(float)
    sigma_m = d["sigma_m"].astype(float)
    k = d["k"].astype(float)
    sigma_k = d["sigma_k"].astype(float)
    mu = d["mu"].astype(float) if "mu" in d else None
    return m, sigma_m, k, sigma_k, mu, f

def load_posterior(post_dir: Path, idx: int, zmin: float, zmax: float,
                   warmup=10000, samples=10000, chains=4):
    f = post_dir / f"y3_bin{idx}_z{zmin}-{zmax}_all_order1_w{warmup}_s{samples}_c{chains}.npz"
    p = np.load(f)

    ln10 = np.log(10.0)
    if "log_Nphi" in p:
        log_Nphi = p["log_Nphi"]
    else:
        log_Nphi = p["log10_Nphi"] * ln10

    if "log_Lstar" in p:
        log_Lstar = p["log_Lstar"]
    else:
        log_Lstar = p["log10_Lstar"] * ln10

    sigma_coeff = p["sigma_coeff"] if "sigma_coeff" in p else None
    m_ref = float(p["m_ref"]) if "m_ref" in p else 22.0

    out = {
        "m0": p["m0"],
        "b": p["b"],
        "alpha": p["alpha"],
        "beta": p["beta"],
        "log_Nphi": log_Nphi,
        "log_Lstar": log_Lstar,
        "sigma_coeff": sigma_coeff,
        "m_ref": m_ref,
    }
    return out, f

def sigma_m_of_m(m: np.ndarray, sigma_coeff: np.ndarray, m_ref: float) -> np.ndarray:
    x = m - m_ref
    log10_sigma = np.polyval(sigma_coeff, x)
    return np.power(10.0, log10_sigma)

def intensity_log(m, mu_scalar, sigma_coeff, m_ref, k, sigma_k,
                  m0, b, alpha, beta, log_Nphi, log_Lstar, ln10):
    """
    Matches the final order-1 model used in the production MCMC:
      - constant mu(z_center)
      - order-1 sigma_m(m)
      - weighted mean k
      - absorbed Lstar convention (no L_bol0 term)
    """
    c = ln10 / 2.5

    # weighted mean k
    w = 1.0 / (sigma_k ** 2)
    w[~np.isfinite(w)] = 0.0
    k_mean = (k * w).sum() / max(w.sum(), 1e-300)

    sigma_m = sigma_m_of_m(m, sigma_coeff, m_ref)

    M0 = m - mu_scalar - k_mean - c * (sigma_m ** 2) * alpha
    lnL0 = -0.4 * ln10 * M0 - log_Lstar

    log_den = np.logaddexp(
        -(alpha + 1.0) * lnL0,
        -(beta + 1.0) * lnL0
    )

    log_eff = -softplus_stable(b * (m - m0))
    exponent = -0.5 * (c ** 2) * (sigma_m ** 2) * (alpha ** 2)

    return log_Nphi + log_eff + exponent - log_den + np.log(c)

def predict_counts_per_mbin(m, mu_scalar, sigma_coeff, m_ref, k, sigma_k,
                            params, edges, ln10):
    """
    For one posterior draw, return predicted counts per m-bin
    by integrating I(m) over m in each bin using trapz on sorted m.
    """
    logI = intensity_log(
        m=m,
        mu_scalar=mu_scalar,
        sigma_coeff=sigma_coeff,
        m_ref=m_ref,
        k=k,
        sigma_k=sigma_k,
        m0=params["m0"],
        b=params["b"],
        alpha=params["alpha"],
        beta=params["beta"],
        log_Nphi=params["log_Nphi"],
        log_Lstar=params["log_Lstar"],
        ln10=ln10,
    )

    order = np.argsort(m)
    m_sorted = m[order]
    I_sorted = np.exp(logI[order])

    preds = np.zeros(len(edges) - 1, dtype=float)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (m_sorted >= lo) & (m_sorted < hi if i < len(edges) - 2 else m_sorted <= hi)
        if sel.sum() >= 2:
            preds[i] = np.trapz(I_sorted[sel], m_sorted[sel])
        else:
            preds[i] = 0.0
    return preds

def parse_bins(s: str):
    """
    Parse:
      "1:2.3-2.4,2:2.4-2.5,3:2.5-2.6,4:2.6-2.8"
    into a list of (idx, zmin, zmax).
    """
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i_part, z_part = tok.split(":")
        zmin_s, zmax_s = z_part.split("-")
        out.append((int(i_part), float(zmin_s), float(zmax_s)))
    return out

def main():
    home = Path.home()
    default_root = home / "project"

    ap = argparse.ArgumentParser(description="Predict DESI counts per m-bin from fitted QLF.")
    ap.add_argument("--arr-dir", default=str(default_root / "data/processed"))
    ap.add_argument("--post-dir", default=str(default_root / "results/mcmc/per_bin/order1_all"))
    ap.add_argument("--out-table-dir", default=str(default_root / "results/tables"))
    ap.add_argument("--out-fig-dir", default=str(default_root / "results/figures/pred_counts"))
    ap.add_argument("--bins", default="1:2.3-2.4,2:2.4-2.5,3:2.5-2.6,4:2.6-2.8",
                    help='Format: "1:2.3-2.4,2:2.4-2.5,3:2.5-2.6,4:2.6-2.8"')
    ap.add_argument("--m-bins", type=int, default=20)
    ap.add_argument("--draws", type=int, default=400, help="Posterior draws per bin for prediction.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=10000)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    arr_dir = Path(args.arr_dir)
    post_dir = Path(args.post_dir)
    out_tab_dir = Path(args.out_table_dir)
    out_tab_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = Path(args.out_fig_dir)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    bins = parse_bins(args.bins)
    rng = np.random.default_rng(args.seed)
    ln10 = np.log(10.0)

    for idx, zmin, zmax in bins:
        # ---- load data and posterior ----
        m, sigma_m_data, k, sigma_k, mu_arr, arr_file = load_bin_arrays(arr_dir, idx, zmin, zmax)
        post, post_file = load_posterior(
            post_dir, idx, zmin, zmax,
            warmup=args.warmup, samples=args.samples, chains=args.chains
        )

        if post["sigma_coeff"] is None:
            raise RuntimeError(f"[bin {idx}] sigma_coeff not found in posterior file: {post_file}")

        sigma_coeff = np.asarray(post["sigma_coeff"], dtype=float)
        m_ref = float(post["m_ref"])

        # final model uses constant mu at bin center
        mu_scalar = float(Planck18.distmod(0.5 * (zmin + zmax)).value)

        n_post = len(post["m0"])
        draws = rng.choice(n_post, size=min(args.draws, n_post), replace=False)

        # ---- magnitude binning ----
        m_lo, m_hi = float(m.min()), float(m.max())
        edges = np.linspace(m_lo, m_hi, args.m_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        obs_counts, _ = np.histogram(m, bins=edges)

        # ---- posterior expected counts and posterior predictive counts ----
        lambda_mat = np.zeros((len(draws), args.m_bins), dtype=float)
        ppc_mat = np.zeros((len(draws), args.m_bins), dtype=float)
        for j, s in enumerate(draws):
            params = {name: post[name][s] for name in ["m0", "b", "alpha", "beta", "log_Nphi", "log_Lstar"]}
            lam = predict_counts_per_mbin(
                m=m,
                mu_scalar=mu_scalar,
                sigma_coeff=sigma_coeff,
                m_ref=m_ref,
                k=k,
                sigma_k=sigma_k,
                params=params,
                edges=edges,
                ln10=ln10,
            )
            lambda_mat[j] = lam
            ppc_mat[j] = rng.poisson(lam)
        pred_mean = np.nanmean(lambda_mat, axis=0)
        exp_lo, exp_hi = np.nanpercentile(lambda_mat, [16, 84], axis=0)
        pred_lo, pred_hi = np.nanpercentile(ppc_mat, [16, 84], axis=0)

        # ---- save CSV ----
        out_csv = out_tab_dir / f"pred_counts_order1_all_m{args.m_bins}_bin{idx}_z{zmin}-{zmax}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "zmin", "zmax", "m_lo", "m_hi", "m_center",
                "N_obs",
                "N_exp_mean",
                "pred_lo", "pred_hi",
                "exp_lo" , "exp_hi"
            ])
            for i in range(args.m_bins):
                w.writerow([
                    zmin, zmax,
                    edges[i], edges[i + 1], centers[i],
                    int(obs_counts[i]),
                    pred_mean[i], pred_lo[i], pred_hi[i],
                    exp_lo,exp_hi
                ])
        print(f"[bin {idx}] wrote {out_csv}")

        # ---- overlay plot ----
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        width = (edges[1] - edges[0]) * 0.9
        ax.bar(centers, obs_counts, width=width, alpha=0.45, label="DESI observed")
        ax.plot(centers, pred_mean, lw=2, label="Posterior mean expected counts")
        ax.fill_between(centers, pred_lo, pred_hi, alpha=0.20, label="Posterior predictive (16–84%)")
        ax.set_xlabel(r"$m_R$")
        ax.set_ylabel("Counts per mag-bin")
        ax.set_title(f"z={zmin}-{zmax} (m bins={args.m_bins})")
        ax.legend()
        fig.tight_layout()

        out_fig = out_fig_dir / f"pred_counts_order1_all_m{args.m_bins}_bin{idx}_z{zmin}-{zmax}.png"
        fig.savefig(out_fig, dpi=200)
        plt.close(fig)
        print(f"[bin {idx}] wrote {out_fig}")

    print("Done.")

if __name__ == "__main__":
    main()
