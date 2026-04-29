#!/usr/bin/env python3
"""
Build DESI Y3 intermediate FITS with K_R and K_R_ERR + FULL cut-flow tracking,
then FILTER OUT non-finite K_R / K_R_ERR and save as:

    ~/project/data/interm/new_d_y3_clean.fits

This version uses redshift range: 2.3 <= z <= 2.8 (implemented as (2.3, 2.8) in cuts).

Outputs
-------
- ~/project/data/interm/new_d_y3_clean.fits
- ~/project/data/interm/new_d_y3_clean.cutflow.json
"""

import os, re, time, json
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict

import numpy as np
import fitsio
from astropy.table import Table
from tqdm import tqdm

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18

from y3pipe.desi import (
    raw_paths, read_filter, get_ws,
    get_tid_tileids_nights, get_indiv_spectra, k_correction_flux
)

# ---------------- CONFIG ----------------
ROOT            = Path.home() / "project"
RAW_DIR         = ROOT / "data/raw/desi_y3_qso"

SUMMARY_NAME    = "desi-ztf-qso-loa-pernight-summary.fits"
PERNIGHT_DIR    = "pernight-spectra"
PERNIGHT_PREFIX = "desi-ztf-qso-loa"
PERNIGHT_FILE_PREFIX = "desi-ztf-qso-loa"  # passed to get_indiv_spectra()

R_FILTER_LOCAL  = ROOT / "data/raw/filters/decam_r.dat"

# FINAL output: cleaned
OUT_FITS        = ROOT / "data/interm" / "new_d_y3_clean.fits"
CUTFLOW_JSON    = OUT_FITS.with_suffix(".cutflow.json")

# -------- science selection (UPDATED) --------
zmin, zmax           = 2.3, 2.8
band, magmin, magmax = "R", 0.0, 23.0
z0_ref               = 2.3

# performance
PROCS     = 32      # use 4 on login; 12–32 on compute nodes
CHUNKSIZE = 200     # tune 100–1000

# explicit quality cuts
FRACTION_MIN = 0.98
SNR_MIN      = 10.0


# ---------------- small utilities ----------------
def now_iso():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


class CutLogger:
    """Accumulate cut-flow records."""
    def __init__(self):
        self.records = []
        self._t = {}

    def mark_time(self, key: str):
        self._t[key] = time.perf_counter()

    def log(self, step: str, n_rows: int, note: str = "", seconds: float | None = None,
            removed: int | None = None, extra: dict | None = None):
        rec = {"step": step, "n_rows": int(n_rows)}
        if removed is not None:
            rec["removed"] = int(removed)
        if note:
            rec["note"] = note
        if seconds is not None:
            rec["seconds"] = float(seconds)
        if extra:
            rec.update(extra)
        self.records.append(rec)

    def log_delta(self, step: str, before: int, after: int, note: str = "", tkey: str | None = None,
                  extra: dict | None = None):
        sec = None
        if tkey is not None and tkey in self._t:
            sec = time.perf_counter() - self._t[tkey]
        self.log(step, after, note=note, seconds=sec, removed=(before - after), extra=extra)


# ---------------- helpers ----------------
def load_r_band():
    """Read DECam r transmission curve (lambda, throughput)."""
    return read_filter(R_FILTER_LOCAL)


def available_pairs(pernight_dir, prefix):
    """Return set of (TILEID, NIGHT) pairs for which per-night FITS exists."""
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
    Keep targets for which *all observed* (TILEID,NIGHT) pairs exist on disk.
    Works on a Table where multiple rows may correspond to one TARGETID.
    """
    tids   = np.asarray(d["TARGETID"])
    tiles  = np.asarray(d["TILEID"])
    nights = np.asarray(d["NIGHT"])

    idx_by_tid = defaultdict(list)
    for i, tid in enumerate(tids):
        idx_by_tid[int(tid)].append(i)

    keep = np.zeros(len(d), dtype=bool)
    for tid, idxs in idx_by_tid.items():
        observed_pairs = set(zip(tiles[idxs], nights[idxs]))
        ok = all((int(t), int(n)) in pairs_set for t, n in observed_pairs)
        if ok:
            keep[idxs] = True
    return keep


def read_data_with_cutflow(path, zmin, zmax, band, magmin, magmax, clog: CutLogger):
    """
    Read summary FITS and apply staged cuts (dedupe, footprint, z, mag),
    logging each step into clog. Returns the filtered Table.
    """
    file = str(path)
    d = Table(fitsio.read(file, "FIBERMAP"))
    hpxd = Table(fitsio.read(file, "HPXMASTER"))
    assert np.all(d["TARGETID"] == hpxd["TARGETID"])

    clog.log("raw_read", len(d), note="Loaded FIBERMAP and HPXMASTER")

    # merge cols
    for key in hpxd.colnames:
        if key not in d.colnames:
            d[key] = hpxd[key]
    clog.log("merge_hpxmaster", len(d), note="Merged missing columns from HPXMASTER")

    # mags + errors
    with np.errstate(divide="ignore", invalid="ignore"):
        d["APP_MAG_G"] = 22.5 - 2.5 * np.log10(d["FLUX_G"]) - 3.214 * d["EBV"]
        d["APP_MAG_R"] = 22.5 - 2.5 * np.log10(d["FLUX_R"]) - 2.165 * d["EBV"]
        d["APP_MAGERR_G"] = 2.5 / np.log(10) / (d["FLUX_G"] * np.sqrt(d["FLUX_IVAR_G"]))
        d["APP_MAGERR_R"] = 2.5 / np.log(10) / (d["FLUX_R"] * np.sqrt(d["FLUX_IVAR_R"]))
    clog.log("compute_mags", len(d), note="Computed APP_MAG_{G,R} and APP_MAGERR_{G,R}")

    # dedupe by TARGETID (keep first)
    before = len(d)
    _, ii = np.unique(d["TARGETID"], return_index=True)
    ii.sort()
    d = d[ii]
    clog.log_delta("dedupe_targetid", before, len(d), note="Keep first occurrence per TARGETID")

    # footprint cut
    before = len(d)
    ra2 = np.asarray(d["TARGET_RA"], dtype=float).copy()
    ra2[ra2 > 300] -= 360
    cs = SkyCoord(ra=ra2 * u.deg, dec=np.asarray(d["TARGET_DEC"], float) * u.deg, frame="icrs")
    b = cs.galactic.b.value
    dec = np.asarray(d["TARGET_DEC"], float)
    ngc = (b > 0) & (dec < 17)
    sgc = (b < 0) & (dec > -7 - 0.1 * ra2) & (dec < 10 + 0.1 * ra2)
    d = d[ngc | sgc]
    clog.log_delta("footprint_ngc_sgc", before, len(d), note="Simplified DR2-like NGC/SGC footprint")

    # z cut
    before = len(d)
    sel = (d["Z"] > zmin) & (d["Z"] < zmax)
    d = d[sel]
    clog.log_delta("z_cut", before, len(d), note=f"Z in ({zmin}, {zmax})")

    # mag cut
    before = len(d)
    mag_col = f"APP_MAG_{band.upper()}"
    sel = (d[mag_col] > magmin) & (d[mag_col] < magmax)
    d = d[sel]
    clog.log_delta("mag_cut", before, len(d), note=f"{mag_col} in ({magmin}, {magmax})")

    # distance modulus
    d["MU"] = Planck18.distmod(d["Z"]).value
    clog.log("distance_modulus", len(d), note="Computed MU = Planck18.distmod(Z)")

    return d


# ---------------- per-target quality and K-correction ----------------
def fraction_rband(i, d, pernight_dir, nwave, ws_r, bp_r, ws, frac_min=FRACTION_MIN):
    """
    Returns index i if target passes fraction cut; otherwise returns None.

    NOTE: FRACTION_MIN cut is enforced here.
    """
    targetid = d["TARGETID"][i]
    z = d["Z"][i]

    tileids, nights = get_tid_tileids_nights(
        targetid, d["TARGETID"], d["TILEID"], d["NIGHT"]
    )

    fraction_vals = []
    fs_all, ivs_all = get_indiv_spectra(
        targetid, tileids, nights, pernight_dir, nwave, file_prefix=PERNIGHT_FILE_PREFIX
    )

    rf_ws_min = 1600 * (1 + z)
    rf_ws_max = 1850 * (1 + z)

    # Keep the same mask logic you had before
    mask = ((ws >= ws_r.min()) & (ws <= ws_r.max())) | ((ws >= rf_ws_min) & (ws <= rf_ws_max))

    for fs, ivs in zip(fs_all, ivs_all):
        fs = fs[mask]
        ivs = ivs[mask]
        sel = (ivs != 0)
        if sel.sum() == 0:
            continue
        fraction_vals.append(sel.sum() / fs.shape[0])

    frac = float(np.mean(fraction_vals)) if len(fraction_vals) else 0.0
    return i if (frac >= frac_min) else None


def compute_kcorr(i, d, pernight_dir, nwave, ws_r, bp_r, ws, snr_min=SNR_MIN, z0=z0_ref):
    """
    Compute exposure-level K corrections with an SNR cut; combine via IVAR weights.
    Returns (k, kerr). If no usable exposures: (np.nan, np.inf)
    """
    targetid = d["TARGETID"][i]
    z = d["Z"][i]

    tileids, nights = get_tid_tileids_nights(
        targetid, d["TARGETID"], d["TILEID"], d["NIGHT"]
    )

    k_vals, w_vals = [], []
    fs_all, ivs_all = get_indiv_spectra(
        targetid, tileids, nights, pernight_dir, nwave, file_prefix=PERNIGHT_FILE_PREFIX
    )

    for fs, ivs in zip(fs_all, ivs_all):
        sel = (ivs != 0)
        if sel.sum() == 0:
            continue

        fs_sel = fs[sel]
        ivs_sel = ivs[sel]

        snr = np.sum(fs_sel) / np.sqrt(np.sum(1.0 / ivs_sel))
        if snr <= snr_min:
            continue

        k_r, err_r = k_correction_flux(ws[sel], fs_sel, ivs_sel, ws_r, bp_r, z0, z)
        if (not np.isfinite(k_r)) or (not np.isfinite(err_r)) or (err_r == 0):
            continue

        k_vals.append(float(k_r))
        w_vals.append(float(1.0 / err_r**2))

    if len(k_vals) == 0 or np.sum(w_vals) == 0:
        return np.nan, np.inf

    w = np.asarray(w_vals)
    k = np.asarray(k_vals)
    k_mean = float(np.sum(k * w) / np.sum(w))
    k_err  = float(np.sqrt(1.0 / np.sum(w)))
    return k_mean, k_err


# ---------------- main pipeline ----------------
def main():
    # keep BLAS/OpenMP from oversubscribing cores
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t0 = time.perf_counter()
    clog = CutLogger()

    # Paths & filter
    summary, pernight = raw_paths(RAW_DIR, SUMMARY_NAME, PERNIGHT_DIR)
    ws_r, bp_r = load_r_band()

    clog.log(
        "config",
        0,
        note="Configuration recorded",
        extra={
            "timestamp": now_iso(),
            "summary": str(summary),
            "pernight": str(pernight),
            "out_fits": str(OUT_FITS),
            "zmin": zmin, "zmax": zmax,
            "band": band, "magmin": magmin, "magmax": magmax,
            "z0_ref": z0_ref,
            "fraction_min": FRACTION_MIN,
            "snr_min": SNR_MIN,
            "procs": PROCS,
            "chunksize": CHUNKSIZE,
        }
    )

    # --- read and stage cuts inside read_data_with_cutflow ---
    clog.mark_time("read_data_total")
    d = read_data_with_cutflow(summary, zmin, zmax, band, magmin, magmax, clog)
    clog.log("read_data_done", len(d), seconds=(time.perf_counter() - clog._t["read_data_total"]),
             note="Completed read_data_with_cutflow()")

    # --- quick prefilters (positive flux/ivar) ---
    before = len(d)
    sel = (d["FLUX_G"] > 0) & (d["FLUX_R"] > 0) & (d["FLUX_IVAR_G"] > 0) & (d["FLUX_IVAR_R"] > 0)
    d = d[sel]
    clog.log_delta("positive_flux_ivar", before, len(d),
                   note="Require FLUX_G,R>0 and FLUX_IVAR_G,R>0")
    print(f"After quick prefilters: {len(d)}")

    # --- wavelength grid ---
    ws = get_ws(pernight, prefix=PERNIGHT_PREFIX)
    nwave = len(ws)
    clog.log("wavegrid", len(d), note=f"Loaded DESI wavelength grid (nwave={nwave})")

    # --- availability check ---
    clog.mark_time("availability")
    before = len(d)
    pairs = available_pairs(pernight, PERNIGHT_PREFIX)
    keep = mask_targets_with_files(d, pairs)
    d = d[keep]
    clog.log_delta("pernight_availability", before, len(d), tkey="availability",
                   note="Require all observed (TILEID,NIGHT) per-night spectra files exist",
                   extra={"n_pairs_available": int(len(pairs))})
    print(f"Availability kept: {len(d)}")

    # --- fraction filter (explicit FRACTION_MIN) ---
    args1 = [(i, d, pernight, nwave, ws_r, bp_r, ws, FRACTION_MIN) for i in range(len(d))]
    clog.mark_time("fraction_filter")
    with Pool(processes=PROCS) as pool:
        idx_list = list(
            tqdm(
                pool.starmap(fraction_rband, args1, chunksize=CHUNKSIZE),
                total=len(args1),
                desc=f"Filtering fraction (>= {FRACTION_MIN})",
                mininterval=1.0,
            )
        )
    valid_indices = np.array([h for h in idx_list if h is not None], dtype=int)

    before = len(d)
    d = d[valid_indices]
    clog.log_delta("fraction_filter", before, len(d), tkey="fraction_filter",
                   note=f"Keep targets with mean valid-pixel fraction >= {FRACTION_MIN}")
    print(f"Fraction filter kept {len(d)} / {before}")

    # --- k-correction computation (explicit SNR_MIN) ---
    args2 = [(i, d, pernight, nwave, ws_r, bp_r, ws, SNR_MIN, z0_ref) for i in range(len(d))]
    clog.mark_time("kcorr")
    with Pool(processes=PROCS) as pool:
        results = list(
            tqdm(
                pool.starmap(compute_kcorr, args2, chunksize=CHUNKSIZE),
                total=len(args2),
                desc=f"K-correction (SNR>{SNR_MIN})",
                mininterval=1.0,
            )
        )
    clog.log("kcorr_done", len(d), seconds=(time.perf_counter() - clog._t["kcorr"]),
             note="Computed per-target (K_R, K_R_ERR)")

    k_rr, k_rr_err = zip(*results) if results else ([], [])
    d["K_R"]     = np.asarray(k_rr, dtype=float)
    d["K_R_ERR"] = np.asarray(k_rr_err, dtype=float)

    # --- FINAL CLEAN: filter out non-finite K ---
    before = len(d)
    finite = np.isfinite(d["K_R"]) & np.isfinite(d["K_R_ERR"]) & (d["K_R_ERR"] > 0)
    d = d[finite]
    clog.log_delta("finite_kcorr_cut", before, len(d),
                   note="Require finite K_R, finite K_R_ERR, and K_R_ERR > 0")

    # --- write cleaned output FITS ---
    OUT_FITS.parent.mkdir(parents=True, exist_ok=True)
    clog.mark_time("write_fits")
    d.write(OUT_FITS, format="fits", overwrite=True)
    clog.log("write_fits", len(d), seconds=(time.perf_counter() - clog._t["write_fits"]),
             note=f"Wrote {OUT_FITS.name}")

    total = time.perf_counter() - t0
    clog.log("total_runtime", len(d), seconds=total, note="Total pipeline runtime (seconds)")
    print(f"Wrote {OUT_FITS} rows: {len(d)} total: {total:.2f}s")

    # --- write cutflow JSON ---
    payload = {
        "created_at": now_iso(),
        "inputs": {
            "summary": str(summary),
            "pernight": str(pernight),
            "r_filter": str(R_FILTER_LOCAL),
        },
        "outputs": {
            "fits": str(OUT_FITS),
            "cutflow_json": str(CUTFLOW_JSON),
        },
        "cutflow": clog.records,
    }
    with open(CUTFLOW_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved cutflow:", CUTFLOW_JSON)


if __name__ == "__main__":
    main()