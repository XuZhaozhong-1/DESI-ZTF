#!/usr/bin/env python3
"""
Redshift distribution of DESI Y3 quasars
after duplicate removal and footprint selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import fitsio
from numpy.lib import recfunctions as rfn

from astropy.coordinates import SkyCoord
import astropy.units as u

# ============================================================
# Paths
# ============================================================
ROOT = Path.home() / "project"
RAW  = ROOT / "data/raw/desi_y3_qso"
FIBERMAP = RAW / "desi-ztf-qso-loa-pernight-summary.fits"

OUTDIR = ROOT / "results/figures/interm"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFIG = OUTDIR / "z_hist_after_dup_footprint.png"

# ============================================================
# Load raw DESI tables
# ============================================================
print(f"[INFO] Reading {FIBERMAP}")
d   = fitsio.read(FIBERMAP, "FIBERMAP")
hpx = fitsio.read(FIBERMAP, "HPXMASTER")

# Merge HPXMASTER columns
for name in hpx.dtype.names:
    if name not in d.dtype.names:
        d = rfn.append_fields(d, name, hpx[name], usemask=False)

print(f"[INFO] Raw rows: {len(d)}")

# ============================================================
# Duplicate removal (unique TARGETID)
# ============================================================
_, idx = np.unique(d["TARGETID"], return_index=True)
idx.sort()
d = d[idx]

print(f"[INFO] After duplicate removal: {len(d)}")

# ============================================================
# Footprint selection (same logic as pipeline)
# ============================================================
ra2 = d["TARGET_RA"].astype(float)
ra2 = np.where(ra2 > 300.0, ra2 - 360.0, ra2)
dec = d["TARGET_DEC"].astype(float)

cs = SkyCoord(ra=ra2 * u.deg, dec=dec * u.deg, frame="icrs")
b  = cs.galactic.b.value

ngc = (b > 0) & (dec < 17)
sgc = (b < 0) & (dec > -7 - 0.1 * ra2) & (dec < 10 + 0.1 * ra2)

d = d[ngc | sgc]

print(f"[INFO] After footprint selection: {len(d)}")

# ============================================================
# Redshift histogram
# ============================================================
z = d["Z"]

bins = np.linspace(0.0, 5.0, 100)

plt.figure(figsize=(8, 5))
plt.hist(z, bins=bins, histtype="step", lw=2)
plt.xlabel("Redshift $z$", fontsize=13)
plt.ylabel("Number of quasars", fontsize=13)
plt.title(
    "DESI Y3 quasar redshift distribution\n"
    "after duplicate removal and footprint selection"
)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTFIG, dpi=200)
plt.close()

print(f"[INFO] Saved figure to {OUTFIG}")