#!/usr/bin/env python3
"""
Filter out rows with non-finite K_R or K_R_ERR from DESI Y3 intermediate table.
"""

from pathlib import Path
import numpy as np
from astropy.table import Table

# ---------- paths ----------
ROOT     = Path.home() / "project"
IN_FITS  = ROOT / "data/interm/new_d_y3.fits"
OUT_FITS = ROOT / "data/interm/new_d_y3_clean.fits"
BAD_CSV  = ROOT / "results/tables/bad_k_targets.csv"

# ---------- load ----------
print(f"Reading {IN_FITS}")
t = Table.read(IN_FITS, format="fits")

# Convert to plain NumPy arrays to avoid MaskedColumn gotchas
K    = np.array(t["K_R"])
Kerr = np.array(t["K_R_ERR"])

# Define "bad" as nan or inf in K or K_ERR
bad = np.isnan(K) | np.isinf(K) | np.isnan(Kerr) | np.isinf(Kerr)
n_bad = int(bad.sum())

print(f"Total rows: {len(t)}")
print(f"Bad rows (nan/inf in K_R or K_R_ERR): {n_bad}")

if n_bad > 0:
    # Save bad rows for traceability
    bad_tbl = Table({
        "TARGETID": t["TARGETID"][bad],
        "K_R":      t["K_R"][bad],
        "K_R_ERR":  t["K_R_ERR"][bad],
        "Z":        t["Z"][bad],
    })
    BAD_CSV.parent.mkdir(parents=True, exist_ok=True)
    bad_tbl.write(BAD_CSV, format="ascii.csv", overwrite=True)
    print(f"Wrote details of bad rows to {BAD_CSV}")

# Filter & save cleaned table
t_clean = t[~bad]
t_clean.write(OUT_FITS, format="fits", overwrite=True)

print(f"Kept rows: {len(t_clean)}")
print(f"Wrote clean FITS: {OUT_FITS}")
