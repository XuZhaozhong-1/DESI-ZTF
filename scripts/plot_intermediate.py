#!/usr/bin/env python3
# Plots for DESI Y3 intermediate table:
#  - histograms of K_R and K_R_ERR
#  - K_R vs Z (with per-point error bars and running median with 16-84% band)
#  - (m - mu) vs Z
#  - (m - mu - K) vs Z
#  - dispersion comparison in Z-bins (std before/after K)
#  - slope panels with unweighted and weighted fits

from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

# ---------- paths ----------
ROOT    = Path.home() / "project"
IN_FITS = ROOT / "data/interm/new_d_y3_clean.fits"
OUTDIR  = ROOT / "results/figures/interm"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def running_quantiles(x, y, nbins=25, q=(0.16, 0.5, 0.84)):
    bins = np.linspace(x.min(), x.max(), nbins + 1)
    idx  = np.digitize(x, bins) - 1
    xc   = 0.5 * (bins[:-1] + bins[1:])
    qs   = []
    for qq in q:
        v = np.empty(nbins)
        v[:] = np.nan
        for i in range(nbins):
            sel = (idx == i)
            if sel.any():
                v[i] = np.quantile(y[sel], qq)
        qs.append(v)
    return xc, qs

def binned_std(x, y, nbins=12):
    bins = np.linspace(x.min(), x.max(), nbins + 1)
    idx  = np.digitize(x, bins) - 1
    xc   = 0.5 * (bins[:-1] + bins[1:])
    s    = np.empty(nbins)
    s[:] = np.nan
    for i in range(nbins):
        sel = (idx == i)
        if sel.sum() >= 3:
            s[i] = np.std(y[sel])
    return xc, s

def linear_slope(x, y):
    a, b = np.polyfit(x, y, 1)
    return a, b

def linear_slope_weighted(x, y, sigma):
    w = 1.0 / sigma
    a, b = np.polyfit(x, y, 1, w=w)
    return a, b

# ---------- load ----------
print(f"Reading {IN_FITS}")
t = Table.read(IN_FITS, format="fits")

Z     = np.array(t["Z"], dtype=float)
mR    = np.array(t["APP_MAG_R"], dtype=float)
MU    = np.array(t["MU"], dtype=float)
K     = np.array(t["K_R"], dtype=float)
Kerr  = np.array(t["K_R_ERR"], dtype=float)
mErr  = np.array(t["APP_MAGERR_R"], dtype=float)

m_minus_mu   = mR - MU
m_minus_mu_k = mR - MU - K

# ---------- 1) histograms ----------
plt.figure(figsize=(6,4))
plt.hist(K, bins=60, histtype="step")
plt.xlabel("K_R")
plt.ylabel("Count")
plt.title("Histogram of K_R")
plt.tight_layout()
plt.savefig(OUTDIR / "hist_K_R.png", dpi=200)

plt.figure(figsize=(6,4))
plt.hist(Kerr, bins=60, histtype="step")
plt.xlabel("sigma(K_R)")
plt.ylabel("Count")
plt.title("Histogram of K_R uncertainty")
plt.tight_layout()
plt.savefig(OUTDIR / "hist_K_R_ERR.png", dpi=200)

# ---------- 2) K vs Z with error bars and running median ----------
plt.figure(figsize=(6,4))

plt.errorbar(
    Z, K,
    yerr=Kerr,
    fmt=".", ms=1, lw=0,
    alpha=0.15,
    ecolor="0.7",
    elinewidth=0.5,
    capsize=0,
    label="K_R ± σ(K_R)"
)

plt.xlabel("z")
plt.ylabel("K_R")
plt.title("K_R vs z")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "K_R_vs_z.png", dpi=200)

# ---------- 3) (m - mu) vs Z ----------
xc1, (p16a, p50a, p84a) = running_quantiles(Z, m_minus_mu, nbins=25, q=(0.16, 0.5, 0.84))
plt.figure(figsize=(6,4))
plt.scatter(Z, m_minus_mu, s=1, alpha=0.15, label="points")
plt.plot(xc1, p50a, lw=2, label="median")
plt.fill_between(xc1, p16a, p84a, alpha=0.15, label="16-84%")
plt.xlabel("z")
plt.ylabel("m_R - mu")
plt.title("m_R - mu vs z")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "m_minus_mu_vs_z.png", dpi=200)

# ---------- 4) (m - mu - K) vs Z ----------
xc2, (p16b, p50b, p84b) = running_quantiles(Z, m_minus_mu_k, nbins=25, q=(0.16, 0.5, 0.84))
plt.figure(figsize=(6,4))
plt.scatter(Z, m_minus_mu_k, s=1, alpha=0.15, label="points")
plt.plot(xc2, p50b, lw=2, label="median")
plt.fill_between(xc2, p16b, p84b, alpha=0.15, label="16-84%")
plt.xlabel("z")
plt.ylabel("m_R - mu - K_R")
plt.title("m_R - mu - K_R vs z")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "m_minus_mu_minus_K_vs_z.png", dpi=200)

# ---------- 5) dispersion comparison in z-bins ----------
bin_z, std_noK   = binned_std(Z, m_minus_mu,   nbins=12)
_,     std_withK = binned_std(Z, m_minus_mu_k, nbins=12)

plt.figure(figsize=(6,4))
plt.plot(bin_z, std_noK,   marker="o", label="std(m_R - mu)")
plt.plot(bin_z, std_withK, marker="o", label="std(m_R - mu - K_R)")
plt.xlabel("z (bin centers)")
plt.ylabel("Std per bin")
plt.title("Dispersion vs redshift")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "dispersion_vs_z.png", dpi=200)

# ---------- 6) slope panels: unweighted and weighted ----------
slope_noK,   b_noK   = linear_slope(Z, m_minus_mu)
slope_withK, b_withK = linear_slope(Z, m_minus_mu_k)

sigma_withK = np.sqrt(mErr**2 + Kerr**2)
slope_noK_w,   b_noK_w   = linear_slope_weighted(Z, m_minus_mu,   sigma=mErr)
slope_withK_w, b_withK_w = linear_slope_weighted(Z, m_minus_mu_k, sigma=sigma_withK)

print(f"Slope (unweighted)  m-mu vs z:     {slope_noK:.6f}")
print(f"Slope (weighted)    m-mu vs z:     {slope_noK_w:.6f}")
print(f"Slope (unweighted)  m-mu-K vs z:   {slope_withK:.6f}")
print(f"Slope (weighted)    m-mu-K vs z:   {slope_withK_w:.6f}")

# m - mu panel
zline = np.linspace(Z.min(), Z.max(), 200)
y_unw = slope_noK * zline + b_noK
y_w   = slope_noK_w * zline + b_noK_w

plt.figure(figsize=(6,4))
plt.scatter(Z, m_minus_mu, s=1, alpha=0.15, label="data")
plt.plot(zline, y_unw, lw=2, label=f"unweighted slope={slope_noK:.5f}")
plt.plot(zline, y_w,   lw=2, linestyle="--", color="tab:orange",
         label=f"weighted slope={slope_noK_w:.5f}")
plt.xlabel("z"); plt.ylabel("m_R - mu")
plt.title("m_R - mu linear trend")
plt.legend(); plt.tight_layout()
plt.savefig(OUTDIR / "m_minus_mu_slope.png", dpi=200)

# m - mu - K panel
y_unw = slope_withK * zline + b_withK
y_w   = slope_withK_w * zline + b_withK_w

plt.figure(figsize=(6,4))
plt.scatter(Z, m_minus_mu_k, s=1, alpha=0.15, label="data")
plt.plot(zline, y_unw, lw=2, label=f"unweighted slope={slope_withK:.5f}")
plt.plot(zline, y_w,   lw=2, linestyle="--", color="tab:orange",
         label=f"weighted slope={slope_withK_w:.5f}")
plt.xlabel("z"); plt.ylabel("m_R - mu - K_R")
plt.title("m_R - mu - K_R linear trend")
plt.legend(); plt.tight_layout()
plt.savefig(OUTDIR / "m_minus_mu_minus_K_slope.png", dpi=200)

print(f"Saved figures to: {OUTDIR}")
