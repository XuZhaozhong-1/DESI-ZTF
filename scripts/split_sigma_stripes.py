#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

ROOT = Path.home() / "project"
DATA = ROOT / "data" / "processed"
OUT  = DATA / "splits"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Bin definitions (match filenames)
# --------------------------------------------------
BINS = {
    1: (2.3, 2.4),
    2: (2.4, 2.5),
    3: (2.5, 2.6),
    4: (2.6, 2.8),
}

# --------------------------------------------------
# Stripe splitting via quantile ridges
# --------------------------------------------------
def split_stripes(
    m,
    sigma_m,
    q_upper=0.90,
    q_lower=0.10,
    nbins=40,
    min_per_bin=80,
    poly_deg=1,
    mid_w_upper=0.25,
    mid_w_lower=0.75,
):
    """
    Split points into upper / lower stripes in log10(sigma_m) vs m.
    Returns boolean masks: all, upper, lower
    """

    m = np.asarray(m)
    sig = np.asarray(sigma_m)

    m0 = m
    y0 = np.log10(sig)

    edges = np.linspace(m0.min(), m0.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    yq_u = np.full(nbins, np.nan)
    yq_l = np.full(nbins, np.nan)

    for i in range(nbins):
        sel = (m0 >= edges[i]) & (m0 < edges[i + 1])
        if sel.sum() < min_per_bin:
            continue
        yq_u[i] = np.quantile(y0[sel], q_upper)
        yq_l[i] = np.quantile(y0[sel], q_lower)

    good = np.isfinite(yq_u) & np.isfinite(yq_l)
    if good.sum() < max(6, poly_deg + 3):
        raise RuntimeError("Not enough bins to fit ridge")

    xc = centers[good]
    yu = yq_u[good]
    yl = yq_l[good]

    p_upper = np.polyfit(xc, yu, deg=poly_deg)
    p_lower = np.polyfit(xc, yl, deg=poly_deg)

    def ridge_upper(mm): return np.polyval(p_upper, mm)
    def ridge_lower(mm): return np.polyval(p_lower, mm)
    def midline(mm):     return mid_w_upper * ridge_upper(mm) + mid_w_lower * ridge_lower(mm)

    up_local = y0 >= midline(m0)
    lo_local = ~up_local

    upper = np.zeros_like(m, dtype=bool)
    lower = np.zeros_like(m, dtype=bool)
    upper = up_local
    lower = lo_local

    all_mask = np.ones_like(m, dtype=bool)

    return all_mask, upper, lower, (p_upper, p_lower)

# --------------------------------------------------
# Plot helper
# --------------------------------------------------
def save_split_plot(tag, m, sigma_m, upper, lower, p_upper, p_lower):
    mm = m
    yy = np.log10(sigma_m)

    mline = np.linspace(mm.min(), mm.max(), 400)
    ru = np.polyval(p_upper, mline)
    rl = np.polyval(p_lower, mline)
    mid = 0.25 * ru + 0.75 * rl

    plt.figure(figsize=(8, 5.5))
    plt.scatter(m[lower], np.log10(sigma_m[lower]), s=2, alpha=0.25, label="Lower stripe")
    plt.scatter(m[upper], np.log10(sigma_m[upper]), s=2, alpha=0.25, label="Upper stripe")
    plt.plot(mline, ru, lw=2, label="Upper ridge")
    plt.plot(mline, rl, lw=2, label="Lower ridge")
    plt.plot(mline, mid, lw=2, ls="--", label="Midline")
    plt.xlabel("m")
    plt.ylabel(r"$\log_{10}\sigma_m$")
    plt.title(tag)
    plt.legend(markerscale=4)
    plt.tight_layout()

    outpng = OUT / f"fig_{tag}_split.png"
    plt.savefig(outpng, dpi=200)
    plt.close()
    print("  wrote", outpng.name)

def update_counts_json(json_path, tag, all_mask, upper, lower):
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[tag] = {
        "N_all": int(all_mask.sum()),
        "N_upper": int(upper.sum()),
        "N_lower": int(lower.sum()),
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("  updated", json_path.name)
# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    counts_json = OUT / "split_counts.json"
    for idx, (zmin, zmax) in BINS.items():
        tag = f"bin{idx}_z{zmin}-{zmax}"
        npz = DATA / f"y3_bin{idx}_z{zmin}-{zmax}_arrays.npz"

        print(f"[{tag}] loading {npz.name}")
        d = np.load(npz)
        m = d["apr_mag_r"]
        sigma_m = d["sigma_m"]

        all_mask, upper, lower, (p_up, p_lo) = split_stripes(m, sigma_m)

        out_npz = OUT / f"{tag}_split.npz"
        np.savez(
            out_npz,
            all=all_mask,
            upper=upper,
            lower=lower,
        )
        print("  wrote", out_npz.name)
        print(f"  N(all)={all_mask.sum():,} N(upper)={upper.sum():,} N(lower)={lower.sum():,}")
        update_counts_json(
            counts_json,
            tag,
            all_mask,
            upper,
            lower,
        )
        save_split_plot(tag, m, sigma_m, upper, lower, p_up, p_lo)

    print("\nDone.")

if __name__ == "__main__":
    main()
