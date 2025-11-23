#!/usr/bin/env python3
"""
Rest-frame dispersion of DESI Y3 quasar spectra, using RAW data only.

This reproduces your previous ZTF-style dispersion figure,
but uses DESI per-night BRZ_FLUX/IVAR without any K-correction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from astropy.cosmology import Planck18
from y3pipe.desi import (
    read_data,
    get_ws,
    get_tid_tileids_nights,
    get_indiv_spectra,
)

# =====================================================================
# CONFIG
# =====================================================================
ROOT = Path.home() / "project"
RAW = ROOT / "data/raw/desi_y3_qso"
PERNIGHT = RAW / "pernight-spectra"
FIBERMAP = RAW / "desi-ztf-qso-loa-pernight-summary.fits"   # adjust if named differently
PREFIX = "desi-ztf-qso-loa"

# Selection cuts (raw)
zmin, zmax = 2.3, 2.8
band = "R"
magmin, magmax = 0, 23

# Dispersion reference redshift
z0 = 2.3
mu_ref = Planck18.distmod(z0).value

# Output
OUTDIR = ROOT / "results/figures/interm"
OUTDIR.mkdir(exist_ok=True, parents=True)
OUTFIG = OUTDIR / "restframe_dispersion_raw_y3.png"

# =====================================================================
# Load raw DESI Y3 target metadata before K-correction
# =====================================================================
print(f"[INFO] Reading raw DESI metadata from {FIBERMAP}")
d = read_data(FIBERMAP, zmin, zmax, band, magmin, magmax)
Nobj = len(d)
print(f"[INFO] Selected {Nobj} objects for raw dispersion")

# DESI wavelength grid (BRZ_WAVE)
ws = get_ws(PERNIGHT, prefix=PREFIX)
nwave = len(ws)
print(f"[INFO] Using DESI per-night wavelength grid N={nwave}, "
      f"{ws.min():.1f}–{ws.max():.1f} Å")

# =====================================================================
# Common observed-frame grid at z0
# =====================================================================
new_ws_obs = np.arange(3600.0, 8011.0, 1.0)
new_ws_rf  = new_ws_obs / (1 + z0)

# =====================================================================
# Per-target processing
# =====================================================================
def process_one(i):
    """Process a single target: load RAW per-night spectra, combine, shift."""
    tid   = int(d["TARGETID"][i])
    z     = float(d["Z"][i])
    mu    = float(d["MU"][i])

    # Find exposures
    tileids, nights = get_tid_tileids_nights(
        tid,
        d["TARGETID"], d["TILEID"], d["NIGHT"]
    )

    # Read per-night spectra
    fs_all, ivs_all = get_indiv_spectra(
        tid,
        tileids,
        nights,
        pernight_dir=PERNIGHT,
        nwave=nwave,
        file_prefix=PREFIX
    )

    fs_all = np.asarray(fs_all)
    ivs_all = np.asarray(ivs_all)

    # Pixels that have nonzero IVAR in all exposures
    good = (ivs_all != 0).all(axis=0)
    if not np.any(good):
        return None

    iv = ivs_all[:, good]
    fl = fs_all[:, good]
    ws_good = ws[good]

    # IVAR-weighted stack
    fs_comb = (fl * iv).sum(axis=0) / iv.sum(axis=0)

    # Shift to reference z0
    fs_z0 = fs_comb * (1 + z) / (1 + z0) \
                    * 10**(mu/2.5) / 10**(mu_ref/2.5)
    ws_z0 = ws_good * (1 + z0) / (1 + z)

    # Interpolate onto common observed-frame grid
    new_fs = np.interp(new_ws_obs, ws_z0, fs_z0, left=np.nan, right=np.nan)

    return new_fs

# =====================================================================
# Run multiprocessing
# =====================================================================
print("[INFO] Building restframe spectrum grid...")
from multiprocessing import Pool

with Pool(32) as pool:
    out = list(tqdm(pool.imap(process_one, range(Nobj)), total=Nobj))

# Drop None objects
spec = np.array([x for x in out if x is not None])
print(f"[INFO] Kept {spec.shape[0]} usable spectra")

# =====================================================================
# Compute dispersion
# =====================================================================
disp = np.nanstd(spec, axis=0)

# =====================================================================
# Plot
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(new_ws_obs, disp, lw=2, color="blue")
ax1.set_xlabel("Observed-frame wavelength at z0 (Å)", fontsize=14)
ax1.set_ylabel("Dispersion (raw flux units)", fontsize=14)
ax1.grid(True)

# Rest-frame axis
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())

ticks = np.linspace(new_ws_obs.min(), new_ws_obs.max(), 8)
ax1.set_xticks(ticks)
ax2.set_xticks(ticks)
ax2.set_xticklabels((ticks / (1 + z0)).astype(int))
ax2.set_xlabel("Rest-frame wavelength (Å)", fontsize=14)

fig.tight_layout()
fig.savefig(OUTFIG, dpi=200)
print(f"[INFO] Saved {OUTFIG}")
