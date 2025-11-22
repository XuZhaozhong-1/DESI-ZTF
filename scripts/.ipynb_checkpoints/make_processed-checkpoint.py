#!/usr/bin/env python3
# Clean DESI Y3 "interim" -> per-bin arrays for MCMC
# Uses exact columns: Z, APR_MAG_R, APP_MAGERR_R, K_R, K_R_ERR, MU

from pathlib import Path
import json
import numpy as np
import pandas as pd

# Try fitsio; fall back to astropy
try:
    import fitsio
    HAS_FITSIO = True
except Exception:
    from astropy.table import Table
    HAS_FITSIO = False

# ---- paths ----
ROOT  = Path.home() / "project"
INTER = ROOT / "data/interim" / "new_d_y3.fits"
PROC  = ROOT / "data/processed"
PROC.mkdir(parents=True, exist_ok=True)

# ---- redshift bins ----
Z_BINS    = [(2.3, 2.4), (2.4, 2.5), (2.5, 2.6), (2.6, 2.8)]
Z_CENTERS = [0.5*(a+b) for a,b in Z_BINS]

# ---- exact FITS column names (your file) ----
COLS = {
    "Z"          : "z",
    "APP_MAG_R"  : "apr_mag_r",
    "APP_MAGERR_R": "sigma_m",
    "K_R"        : "k",
    "K_R_ERR"    : "sigma_k",
    "MU"         : "mu",
}

def read_fits_selected(path: Path, cols: list[str]) -> pd.DataFrame:
    """Read only requested columns; return DataFrame."""
    if HAS_FITSIO:
        with fitsio.FITS(str(path)) as f:
            data = f[1].read(columns=cols)
        df = pd.DataFrame.from_records(data)
    else:
        t = Table.read(str(path), hdu=1)
        df = t[cols].to_pandas()
    return df

def to_native_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Convert big-endian numeric columns to native endianness (fixes pandas masks)."""
    for c in df.columns:
        dt = df[c].dtype
        if dt.kind in "ui f" and getattr(dt, "byteorder", "=") in (">", "!"):
            arr = df[c].to_numpy()
            df[c] = arr.byteswap().newbyteorder()
    return df

def main():
    print(f"Reading {INTER}")
    need = list(COLS.keys())
    df = read_fits_selected(INTER, need)
    df = to_native_inplace(df)
    # rename to standard names
    df = df.rename(columns=COLS).replace([np.inf, -np.inf], np.nan)

    # enforce float dtypes
    for c in ["z","apr_mag_r","sigma_m","k","sigma_k","mu"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column after rename: {c}")
        df[c] = df[c].astype(np.float64, copy=False)

    # quick summary
    print("Columns:", df.columns.tolist())
    print("Rows   :", len(df))
    print(df.dtypes)

    # NumPy arrays for fast masking
    z = df["z"].to_numpy()
    manifest = {"inputs": str(INTER), "bins": [], "outputs": []}

    # per-bin save
    for i, (zmin, zmax) in enumerate(Z_BINS, start=1):
        sel = (z >= zmin) & (z < zmax)
        idx = np.nonzero(sel)[0]
        n = idx.size
        print(f"Bin {i}  z=[{zmin},{zmax})  -> {n} rows")
        if n == 0:
            continue

        sub = df.iloc[idx]
        out = {
            "apr_mag_r" : sub["apr_mag_r"].to_numpy(),
            "sigma_m" : sub["sigma_m"].to_numpy(),
            "k"       : sub["k"].to_numpy(),
            "sigma_k" : sub["sigma_k"].to_numpy(),
            "mu"      : sub["mu"].to_numpy(),
            "z_center": np.array([Z_CENTERS[i-1]], dtype=float),
            "zmin"    : np.array([zmin], dtype=float),
            "zmax"    : np.array([zmax], dtype=float),
        }
        fname = PROC / f"y3_bin{i}_z{zmin}-{zmax}_arrays.npz"
        np.savez(fname, **out)
        print("  wrote", fname)
        manifest["bins"].append({"idx": i, "zmin": zmin, "zmax": zmax, "z_center": Z_CENTERS[i-1], "rows": int(n)})
        manifest["outputs"].append(str(fname))

    # save a tiny manifest next to outputs
    mpath = PROC / "processed_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest:", mpath)

if __name__ == "__main__":
    main()
