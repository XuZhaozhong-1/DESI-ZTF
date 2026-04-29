#!/usr/bin/env python3
"""
Split each processed DESI bin into two 'sigma_m vs m' stripes via quantile-ridge fits.
Writes:
  - masks npz: results/stripes/bin{idx}_z{zmin}-{zmax}_stripes.npz
  - plots png: results/stripes/fig_bin{idx}_..._{all_ridges,split}.png
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.home() / "project"

BINS = {
    1: (2.3, 2.4),
    2: (2.4, 2.5),
    3: (2.5, 2.6),
    4: (2.6, 2.8),
}

def split_stripes_by_quantile_ridges(
    m, sig,
    q_upper=0.90, q_lower=0.10,
    nbins=40, min_per_bin=80,
    poly_deg=1,
    mid_w_upper=0.25, mid_w_lower=0.75,  # midline = w_up*ridge_up + w_lo*ridge_lo
):
    """
    Fits ridges in y=log10(sig) vs m using quantiles per mag bin, then splits by midline.
    Returns masks + fit coeffs.
    """
    m = np.asarray(m)
    sig = np.asarray(sig)

    valid = np.isfinite(m) & np.isfinite(sig) & (sig > 0)
    idx_valid = np.where(valid)[0]
    m0 = m[valid]
    y0 = np.log10(sig[valid])

    edges = np.linspace(m0.min(), m0.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    yq_u = np.full(nbins, np.nan)
    yq_l = np.full(nbins, np.nan)

    for i in range(nbins):
        inb = (m0 >= edges[i]) & (m0 < edges[i+1])
        if inb.sum() < min_per_bin:
            continue
        yq_u[i] = np.quantile(y0[inb], q_upper)
        yq_l[i] = np.quantile(y0[inb], q_lower)

    good = np.isfinite(yq_u) & np.isfinite(yq_l)
    if good.sum() < max(6, poly_deg + 3):
        raise RuntimeError(f"Not enough good bins for ridge fit: good_bins={good.sum()}")

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

    upper_full = np.zeros_like(valid, dtype=bool)
    lower_full = np.zeros_like(valid, dtype=bool)
    upper_full[idx_valid] = up_local
    lower_full[idx_valid] = lo_local

    fits = dict(
        p_upper=p_upper, p_lower=p_lower,
        q_upper=float(q_upper), q_lower=float(q_lower),
        mid_w_upper=float(mid_w_upper), mid_w_lower=float(mid_w_lower),
        poly_deg=int(poly_deg),
    )
    masks = dict(valid=valid, upper=upper_full, lower=lower_full)
    return masks, fits

def plot_and_save(m, sig, masks, fits, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)

    valid = masks["valid"]
    upper = masks["upper"]
    lower = masks["lower"]

    logsig = np.log10(sig[valid])
    mmin = np.nanmin(m[valid])
    mmax = np.nanmax(m[valid])
    m_line = np.linspace(mmin, mmax, 400)

    ru = np.polyval(fits["p_upper"], m_line)
    rl = np.polyval(fits["p_lower"], m_line)
    mid = fits["mid_w_upper"] * ru + fits["mid_w_lower"] * rl

    # Fig 1: all + ridges
    plt.figure(figsize=(8, 5.5))
    plt.scatter(m[valid], np.log10(sig[valid]), s=2, alpha=0.10, label="All (valid)")
    plt.plot(m_line, ru, lw=2, label=f"Upper ridge (q={fits['q_upper']})")
    plt.plot(m_line, rl, lw=2, label=f"Lower ridge (q={fits['q_lower']})")
    plt.plot(m_line, mid, lw=2, ls="--",
             label=f"Midline ({fits['mid_w_upper']:.2f}*up + {fits['mid_w_lower']:.2f}*lo)")
    plt.xlabel("m (apr_mag_r)")
    plt.ylabel("log10(sigma_m)")
    plt.title(f"{tag}: all points + fitted ridges")
    plt.legend(markerscale=4)
    plt.tight_layout()
    f1 = outdir / f"fig_{tag}_all_ridges.png"
    plt.savefig(f1, dpi=200)
    plt.close()

    # Fig 2: split
    plt.figure(figsize=(8, 5.5))
    plt.scatter(m[lower], np.log10(sig[lower]), s=2, alpha=0.20, label="Lower stripe")
    plt.scatter(m[upper], np.log10(sig[upper]), s=2, alpha=0.20, label="Upper stripe")
    plt.plot(m_line, mid, lw=2, ls="--", label="Midline")
    plt.xlabel("m (apr_mag_r)")
    plt.ylabel("log10(sigma_m)")
    plt.title(f"{tag}: split result")
    plt.legend(markerscale=4)
    plt.tight_layout()
    f2 = outdir / f"fig_{tag}_split.png"
    plt.savefig(f2, dpi=200)
    plt.close()

    return f1, f2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=str, default="1,2,3,4", help="comma list, e.g. 1,2,3,4")
    ap.add_argument("--q-upper", type=float, default=0.90)
    ap.add_argument("--q-lower", type=float, default=0.10)
    ap.add_argument("--mid-w-upper", type=float, default=0.25)
    ap.add_argument("--mid-w-lower", type=float, default=0.75)
    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--min-per-bin", type=int, default=80)
    ap.add_argument("--poly-deg", type=int, default=1)
    args = ap.parse_args()

    outdir = ROOT / "results/stripes"
    outdir.mkdir(parents=True, exist_ok=True)

    for b in [int(x) for x in args.bins.split(",") if x.strip()]:
        if b not in BINS:
            raise SystemExit(f"Unknown bin index: {b}")

        zmin, zmax = BINS[b]
        npz_path = ROOT / f"data/processed/y3_bin{b}_z{zmin}-{zmax}_arrays.npz"
        d = np.load(npz_path)

        m = d["apr_mag_r"].astype(np.float64)
        sig = d["sigma_m"].astype(np.float64)

        masks, fits = split_stripes_by_quantile_ridges(
            m, sig,
            q_upper=args.q_upper, q_lower=args.q_lower,
            nbins=args.nbins, min_per_bin=args.min_per_bin,
            poly_deg=args.poly_deg,
            mid_w_upper=args.mid_w_upper, mid_w_lower=args.mid_w_lower,
        )

        tag = f"bin{b}_z{zmin}-{zmax}"
        print(f"[{tag}] valid={masks['valid'].sum():,} upper={masks['upper'].sum():,} lower={masks['lower'].sum():,}")

        # save masks + fits
        out_npz = outdir / f"{tag}_stripes.npz"
        np.savez(
            out_npz,
            valid=masks["valid"],
            upper=masks["upper"],
            lower=masks["lower"],
            p_upper=fits["p_upper"],
            p_lower=fits["p_lower"],
            q_upper=fits["q_upper"],
            q_lower=fits["q_lower"],
            mid_w_upper=fits["mid_w_upper"],
            mid_w_lower=fits["mid_w_lower"],
            poly_deg=fits["poly_deg"],
        )
        print("  wrote", out_npz)

        # plots
        f1, f2 = plot_and_save(m, sig, masks, fits, outdir, tag)
        print("  wrote", f1.name, "and", f2.name)

if __name__ == "__main__":
    main()