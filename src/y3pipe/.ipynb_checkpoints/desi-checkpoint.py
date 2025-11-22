# y3pipe/desi.py
import os
from glob import glob
from pathlib import Path
import numpy as np
import fitsio
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck18

# ---------- Paths ----------
def raw_paths(raw_dir, summary_name="desi-ztf-qso-loa-pernight-summary.fits",
              pernight_dirname="pernight-spectra"):
    raw_dir = Path(raw_dir)
    summary = raw_dir / summary_name
    pernight = raw_dir / pernight_dirname
    return summary, pernight

# ---------- Catalog read / basic cuts ----------
def read_data(path, zmin, zmax, band, magmin, magmax, select_qso=False):
    file = str(path)
    d = Table(fitsio.read(file, "FIBERMAP"))
    hpxd = Table(fitsio.read(file, "HPXMASTER"))
    assert np.all(d["TARGETID"] == hpxd["TARGETID"])

    # merge cols
    for key in hpxd.colnames:
        if key not in d.colnames:
            d[key] = hpxd[key]

    # mags + errors (safeguard divide-by-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["APP_MAG_G"] = 22.5 - 2.5 * np.log10(d["FLUX_G"]) - 3.214 * d["EBV"]
        d["APP_MAG_R"] = 22.5 - 2.5 * np.log10(d["FLUX_R"]) - 2.165 * d["EBV"]
        d["APP_MAGERR_G"] = 2.5 / np.log(10) / (d["FLUX_G"] * np.sqrt(d["FLUX_IVAR_G"]))
        d["APP_MAGERR_R"] = 2.5 / np.log(10) / (d["FLUX_R"] * np.sqrt(d["FLUX_IVAR_R"]))

    # dedupe by TARGETID (keep first)
    _, ii = np.unique(d["TARGETID"], return_index=True)
    ii.sort()
    d = d[ii]

    # DR2-like footprint cut
    ra2 = np.asarray(d["TARGET_RA"], dtype=float).copy()
    ra2[ra2 > 300] -= 360
    cs = SkyCoord(ra=ra2 * u.deg, dec=d["TARGET_DEC"] * u.deg, frame="icrs")
    b = cs.galactic.b.value
    ngc = (b > 0) & (d["TARGET_DEC"] < 17)
    sgc = (b < 0) & (d["TARGET_DEC"] > -7 - 0.1 * ra2) & (d["TARGET_DEC"] < 10 + 0.1 * ra2)
    d = d[ngc | sgc]

    # z + mag cuts
    sel = (d["Z"] > zmin) & (d["Z"] < zmax)
    d = d[sel]
    mag_col = "APP_MAG_%s" % band.upper()
    sel = (d[mag_col] > magmin) & (d[mag_col] < magmax)
    d = d[sel]

    # distance modulus
    d["MU"] = Planck18.distmod(d["Z"]).value
    return d

# ---------- Filter loader ----------
def read_filter(path):
    arr = np.loadtxt(str(path))
    if arr.ndim == 1 or arr.shape[1] == 1:
        raise ValueError("Filter file must be 2 columns: wavelength, throughput.")
    return arr[:, 0].astype(float), arr[:, 1].astype(float)

# ---------- Wavelength grid from a sample per-night file ----------
def get_ws(pernight_dir, prefix="desi-ztf-qso-loa"):
    # Grab one file to read BRZ_WAVE grid
    patt = str(Path(pernight_dir) / ("%s-*-*.fits" % prefix))
    files = sorted(glob(patt))
    if not files:
        raise FileNotFoundError("No per-night files found under %s" % pernight_dir)
    ws = fitsio.read(files[0], "BRZ_WAVE")
    return ws

# ---------- Map TARGETID -> (tileids, nights) ----------
def get_tid_tileids_nights(tid, all_tids, all_tileids, all_nights):
    sel = (all_tids == tid)
    return all_tileids[sel], all_nights[sel]

# ---------- Read individual spectra for one target ----------
def get_indiv_spectra(tid, tileids, nights, pernight_dir, nwave, file_prefix="desi-ztf-qso-loa"):
    nobs = len(tileids)
    fs = np.zeros((nobs, nwave))
    ivs = np.zeros((nobs, nwave))
    for i, (tileid, night) in enumerate(zip(tileids, nights)):
        fn = Path(pernight_dir) / ("%s-%d-%d.fits" % (file_prefix, int(tileid), int(night)))
        tmp_fm = fitsio.read(fn, "FIBERMAP", columns=["TARGETID"])
        tmp_i = np.where(tmp_fm["TARGETID"] == tid)[0][0]
        h = fitsio.FITS(fn)
        tmp_slice = slice(tmp_i, tmp_i + 1, 1)
        fs[i, :] = h["BRZ_FLUX"][tmp_slice, :]
        ivs[i, :] = h["BRZ_IVAR"][tmp_slice, :]
    return fs, ivs

# ---------- K-correction from spectrum ----------
def k_correction_flux(ws, fs, ivs, ws_bp, bp, z0, z, calib_ws=None, calib_fs=None):
    # limit wavelength window
    mask = (ws > 5280.0) & (ws < 7350.0)
    ws = ws[mask]; fs = fs[mask]; ivs = ivs[mask]

    sigma_square = 1.0 / ivs

    # observed-frame bandpass
    bp_ro = np.zeros_like(ws)
    mask_ro = (ws > 5400.0)
    ws_ro = ws[mask_ro]
    bp_ro[mask_ro] = np.interp(ws_ro, ws_bp, bp)

    # rest-frame window mapped to observed
    bp_rf = np.zeros_like(ws)
    mask_rf = (ws > 1600.0*(1.0+z)) & (ws < 1850.0*(1.0+z))
    bp_rf[mask_rf] = 1.0

    # numerical integration
    dlam = np.gradient(ws)
    num = np.sum(dlam * bp_rf * fs)
    denom = np.sum(dlam * bp_ro * fs)

    if not np.isfinite(num) or not np.isfinite(denom) or num <= 0 or denom <= 0:
        return np.nan, np.inf

    k = 2.5 * np.log10(num / denom)
    var = (2.5/np.log(10))**2 / (num**2) * np.sum(dlam**2 * sigma_square * (bp_rf - bp_ro * num / denom)**2)
    sk = np.sqrt(var)
    return k, sk
