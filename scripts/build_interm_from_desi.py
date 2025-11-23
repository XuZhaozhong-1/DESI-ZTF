#!/usr/bin/env python3
# Build "interm" FITS with K_R and K_R_ERR (fast path: no per-row exists(), modest multiprocessing)

import os, re, time
from glob import glob
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astropy.table import Table
from tqdm import tqdm

from y3pipe.desi import (
    raw_paths, read_data, read_filter,
    get_ws, get_tid_tileids_nights, get_indiv_spectra, k_correction_flux
)

# ------- CONFIG -------
ROOT            = Path.home() / "project"
RAW_DIR         = ROOT / "data/raw/desi_y3_qso"
OUT_FITS        = ROOT / "data/interm" / "new_d_y3.fits"

SUMMARY_NAME    = "desi-ztf-qso-loa-pernight-summary.fits"
PERNIGHT_DIR    = "pernight-spectra"
PERNIGHT_PREFIX = "desi-ztf-qso-loa"

R_FILTER_LOCAL  = ROOT / "data/raw/filters/decam_r.dat"

zmin, zmax          = 2.3, 2.8
band, magmin, magmax = "R", 0.0, 23.0
z0_ref              = 2.3
PROCS               = 32  # use 4 on login; 12-16 on a compute node

# ------- helpers -------
def load_r_band():
    return read_filter(R_FILTER_LOCAL)

def available_pairs(pernight_dir, prefix):
    rx = re.compile(r"%s-(\d+)-(\d+)\.fits$" % re.escape(prefix))
    pairs = set()
    patt = str(Path(pernight_dir) / ("%s-*-*.fits" % prefix))
    for fn in glob(patt):
        m = rx.search(fn)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
    return pairs

def mask_targets_with_files(d, pairs_set):
    """
    Return a boolean mask selecting targets for which *all actually observed*
    (TILEID, NIGHT) pairs have corresponding per-night spectra on disk.

    Parameters
    ----------
    d : table-like
        Must have columns TARGETID, TILEID, NIGHT.
    pairs_set : set of (tileid, night)
        Set of (TILEID, NIGHT) pairs for which a per-night FITS file exists,
        as returned by `available_pairs`.

    Returns
    -------
    keep : ndarray of bool, shape (len(d),)
        True for all rows belonging to targets that have complete per-night
        coverage; False otherwise.
    """
    from collections import defaultdict
    tids   = d["TARGETID"]
    tiles  = d["TILEID"]
    nights = d["NIGHT"]

    # Group row indices by TARGETID
    idx_by_tid = defaultdict(list)
    for i, tid in enumerate(tids):
        idx_by_tid[int(tid)].append(i)

    keep = np.zeros(len(d), dtype=bool)

    for tid, idxs in idx_by_tid.items():
        # All actually observed (tile, night) pairs for this target
        observed_pairs = set(zip(tiles[idxs], nights[idxs]))

        # Require that every observed pair exists on disk
        ok = all((int(t), int(n)) in pairs_set for t, n in observed_pairs)

        if ok:
            keep[idxs] = True

    return keep

def fraction_rband(i,d,pernight_dir,nwave,ws_r,bp_r,ws):
    targetid = d['TARGETID'][i]
    z = d['Z'][i]

    tileids, nights = get_tid_tileids_nights(
        targetid,
        d["TARGETID"],
        d["TILEID"],
        d["NIGHT"]
    )

    fraction_vals = []
    fs_all, ivs_all = get_indiv_spectra(targetid, tileids, nights, pernight_dir, nwave, file_prefix="desi-ztf-qso-loa")
    rf_ws_min = 1600 * (1 + z) 
    rf_ws_max = 1850 * (1 + z)
    #mask = (ws >= ws_r.min()) & (ws <= ws_r.max()) & (ws >= rf_ws_min) & (ws <= rf_ws_max)
    mask = ((ws >= ws_r.min()) & (ws <= ws_r.max())) | ((ws >= rf_ws_min) & (ws <= rf_ws_max))

    for fs, ivs in zip(fs_all, ivs_all):
        fs = fs[mask]
        ivs = ivs[mask]

        SEL = (ivs != 0)
        if SEL.sum() == 0:
            continue

        fs_sel = fs[SEL]
        ivs_sel = ivs[SEL]

        fraction = fs_sel.shape[0] / fs.shape[0]

        fraction_vals.append(fraction)

    if len(fraction_vals) == 0:
        fraction = 0
    else:
        fraction = np.mean(fraction_vals)
    if fraction >= 0.98:
        return i
        
def compute_kcorr(i,d,PERNIGHT_DIR,nwave,ws_r,bp_r,ws):
    targetid = d['TARGETID'][i]
    z = d['Z'][i]

    tileids, nights = get_tid_tileids_nights(
        targetid,
        d["TARGETID"],
        d["TILEID"],
        d["NIGHT"]
    )

    k_rr_vals, w_rr_vals = [], []

    fs_all, ivs_all = get_indiv_spectra(targetid, tileids, nights, PERNIGHT_DIR, nwave, file_prefix="desi-ztf-qso-loa")

    for fs, ivs in zip(fs_all, ivs_all):
        fs = fs
        ivs = ivs

        SEL = (ivs != 0)
        if SEL.sum() == 0:
            continue

        fs_sel = fs[SEL]
        ivs_sel = ivs[SEL]

        snr = np.sum(fs_sel) / np.sqrt(np.sum(1/ivs_sel))
        if snr <= 10:
            continue

        k_r, err_r = k_correction_flux(ws[SEL], fs_sel, ivs_sel, ws_r, bp_r, 2.3, z)

        if not np.isfinite(k_r) or not np.isfinite(err_r) or err_r == 0:
            continue

        k_rr_vals.append(k_r)
        w_rr_vals.append(1.0 / err_r**2)

    if len(k_rr_vals) == 0 or np.sum(w_rr_vals) == 0:
        return np.nan, np.inf

    k_rr = np.sum(np.array(k_rr_vals) * np.array(w_rr_vals)) / np.sum(w_rr_vals)
    k_rr_err = np.sqrt(1.0 / np.sum(w_rr_vals))

    return k_rr, k_rr_err



# ------- main -------
def main():
    # keep BLAS/OpenMP from oversubscribing cores
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t0 = time.perf_counter()

    # Paths & filter
    summary, pernight = raw_paths(RAW_DIR, SUMMARY_NAME, PERNIGHT_DIR)
    ws_r, bp_r = load_r_band()  # your helper that loads the DECam r filter

    # Read data & quick prefilters (cheap gates before heavy I/O)
    d = read_data(summary, zmin, zmax, band, magmin, magmax)
    sel = (d["FLUX_G"] > 0) & (d["FLUX_R"] > 0) & (d["FLUX_IVAR_G"] > 0) & (d["FLUX_IVAR_R"] > 0)
    d = d[sel]
    print(f"After quick prefilters: {len(d)}")

    # Wavelength grid from one per-night file
    ws = get_ws(pernight, prefix=PERNIGHT_PREFIX)
    nwave = len(ws)
    # Fast availability check (no per-target os.path.exists in the hot loop)
    ta = time.perf_counter()
    pairs = available_pairs(pernight, PERNIGHT_PREFIX)
    keep = mask_targets_with_files(d, pairs)
    d = d[keep]
    print(f"Availability kept: {len(d)} took {time.perf_counter()-ta:.2f}s")


    # Build arg tuples for starmap
    args1 = [(i,d,pernight,nwave,ws_r,bp_r,ws) for i in range(len(d))]

    # K-corrections
    tb = time.perf_counter()
    CHUNKSIZE = 200  # tune 100–1000 for throughput

    with Pool(processes=PROCS) as pool:
        indices = list(
            tqdm(
                pool.starmap(fraction_rband, args1, chunksize=CHUNKSIZE),
                total=len(args1),
                desc="Filtering fraction",
                mininterval=1.0,
            )
        )
    # keep only integer indices
    valid_indices = np.array([h for h in indices if h is not None], dtype=int)

    print(f"Fraction filter kept {valid_indices.size} / {len(d)} targets")
    d = d[valid_indices]
    args2 = [(i,d,pernight,nwave,ws_r,bp_r,ws) for i in range(len(d))]
    with Pool(processes=PROCS) as pool:
        results = list(
            tqdm(
                pool.starmap(compute_kcorr, args2, chunksize=CHUNKSIZE),
                total=len(args2),
                desc="K-correction r-band",
                mininterval=1.0,
            )
        )
    print(f"K-corr stage: {time.perf_counter()-tb:.2f}s")

    # Attach results and write
    k_rr, k_rr_err = zip(*results) if results else ([], [])
    d["K_R"]     = np.asarray(k_rr, dtype=float)
    d["K_R_ERR"] = np.asarray(k_rr_err, dtype=float)

    OUT_FITS.parent.mkdir(parents=True, exist_ok=True)
    d.write(OUT_FITS, format="fits", overwrite=True)
    print(f"Wrote {OUT_FITS} rows: {len(d)} total: {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()