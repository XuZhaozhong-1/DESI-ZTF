#!/usr/bin/env python3
"""
Make HEALPix density maps before and after the footprint cut
to visually compare angular uniformity.

Outputs (for NSIDE=64):
  ~/project/results/figures/density_before_footprint_nside64_equ.png
  ~/project/results/figures/density_after_footprint_nside64_equ.png
  ~/project/results/figures/density_before_footprint_nside64_gal.png
  ~/project/results/figures/density_after_footprint_nside64_gal.png
"""

from pathlib import Path

import numpy as np
import fitsio
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch runs
import matplotlib.pyplot as plt
import healpy as hp

# ---------- config ----------
ROOT         = Path.home() / "project"
RAW_DIR      = ROOT / "data/raw/desi_y3_qso"
SUMMARY_NAME = "desi-ztf-qso-loa-pernight-summary.fits"

zmin, zmax           = 2.3, 2.8
band, magmin, magmax = "R", 0.0, 23.0

OUTDIR = ROOT / "results/figures/interm"
OUTDIR.mkdir(parents=True, exist_ok=True)

NSIDE = 64  # HEALPix resolution; change if you want finer/coarser maps


# ---------- helpers ----------
def load_and_merge_summary(summary_path: Path) -> Table:
    """
    Read FIBERMAP and HPXMASTER, assert TARGETID alignment, and merge columns.
    """
    file = str(summary_path)
    fm = Table(fitsio.read(file, "FIBERMAP"))
    hpx = Table(fitsio.read(file, "HPXMASTER"))

    assert np.all(fm["TARGETID"] == hpx["TARGETID"])

    # merge extra columns from HPXMASTER into FIBERMAP
    for key in hpx.colnames:
        if key not in fm.colnames:
            fm[key] = hpx[key]

    return fm


def add_magnitudes(tab: Table) -> Table:
    """
    Add APP_MAG_G, APP_MAG_R and APP_MAGERR_G/R columns using DESI flux+EBV.
    """
    d = tab.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        d["APP_MAG_G"] = 22.5 - 2.5 * np.log10(d["FLUX_G"]) - 3.214 * d["EBV"]
        d["APP_MAG_R"] = 22.5 - 2.5 * np.log10(d["FLUX_R"]) - 2.165 * d["EBV"]
        d["APP_MAGERR_G"] = (
            2.5 / np.log(10.0) / (d["FLUX_G"] * np.sqrt(d["FLUX_IVAR_G"]))
        )
        d["APP_MAGERR_R"] = (
            2.5 / np.log(10.0) / (d["FLUX_R"] * np.sqrt(d["FLUX_IVAR_R"]))
        )

    # dedupe by TARGETID (keep first)
    _, ii = np.unique(d["TARGETID"], return_index=True)
    ii.sort()
    return d[ii]


def apply_z_mag_cuts(d: Table) -> Table:
    """
    Apply redshift and R-band magnitude cuts, but NOT the angular footprint.
    """
    sel_z = (d["Z"] > zmin) & (d["Z"] < zmax)
    d = d[sel_z]

    mag_col = f"APP_MAG_{band.upper()}"
    sel_mag = (d[mag_col] > magmin) & (d[mag_col] < magmax)
    d = d[sel_mag]

    # # require positive flux and ivar in g,r to avoid crazy mags
    # sel_flux = (
    #     (d["FLUX_G"] > 0)
    #     & (d["FLUX_R"] > 0)
    #     & (d["FLUX_IVAR_G"] > 0)
    #     & (d["FLUX_IVAR_R"] > 0)
    # )
    # d = d[sel_flux]

    return d


def apply_footprint_cut(d: Table) -> Table:
    """
    Apply the NGC/SGC geometric footprint cut you use in your pipeline.
    """
    ra = np.asarray(d["TARGET_RA"], dtype=float)
    dec = np.asarray(d["TARGET_DEC"], dtype=float)

    # RA remapping used in your code
    ra2 = ra.copy()
    ra2[ra2 > 300.0] -= 360.0

    # Convert to Galactic
    cs = SkyCoord(ra=ra2 * u.deg, dec=dec * u.deg, frame="icrs")
    b = cs.galactic.b.value

    # NGC / SGC masks (your geometry)
    ngc = (b > 0.0) & (dec < 17.0)
    sgc = (b < 0.0) & (dec > (-7.0 - 0.1 * ra2)) & (dec < (10.0 + 0.1 * ra2))

    sel = ngc | sgc
    return d[sel]


def build_healpix_map_equatorial(ra_deg, dec_deg, nside):
    """
    Build a HEALPix count map (equatorial coords) from RA/Dec in degrees.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)

    # HEALPix uses theta = colatitude = 90 - dec, phi = RA, both in radians
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=np.float64)

    pix = hp.ang2pix(nside, theta, phi)
    counts = np.bincount(pix, minlength=npix)
    m[: len(counts)] = counts

    # mark empty pixels as UNSEEN so mollview renders background
    m[m == 0] = hp.UNSEEN
    return m


def build_healpix_map_galactic(ra_deg, dec_deg, nside):
    """
    Build a HEALPix count map in Galactic coordinates (l,b) from RA/Dec in degrees.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)

    # Convert to Galactic
    cs = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    l = cs.l.value
    b = cs.b.value

    # HEALPix: theta = colatitude = 90 - b, phi = l
    theta = np.radians(90.0 - b)
    phi = np.radians(l)

    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=np.float64)

    pix = hp.ang2pix(nside, theta, phi)
    counts = np.bincount(pix, minlength=npix)
    m[: len(counts)] = counts

    m[m == 0] = hp.UNSEEN
    return m


def plot_healpix_density(m, title, outfile):
    """
    Plot a HEALPix density map using mollview and save to file.
    """
    fig = plt.figure(figsize=(8, 5))
    hp.mollview(
        m,
        fig=fig.number,
        title=title,
        unit="counts per pixel",
        norm="hist",
    )
    hp.graticule()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


# ---------- main ----------
def main():
    summary = RAW_DIR / SUMMARY_NAME
    print(f"Reading summary file: {summary}")
    d = load_and_merge_summary(summary)
    print(f"Initial rows (full summary): {len(d)}")

    d = add_magnitudes(d)
    d_before = apply_z_mag_cuts(d)
    print(f"Rows after z+mag+flux cuts (before footprint): {len(d_before)}")

    d_after = apply_footprint_cut(d_before)
    print(f"Rows after footprint cut: {len(d_after)}")

    # Equatorial coordinates
    ra_before = np.asarray(d_before["TARGET_RA"], dtype=float)
    dec_before = np.asarray(d_before["TARGET_DEC"], dtype=float)
    ra_after = np.asarray(d_after["TARGET_RA"], dtype=float)
    dec_after = np.asarray(d_after["TARGET_DEC"], dtype=float)

    # --- HEALPix maps: Equatorial ---
    m_before_equ = build_healpix_map_equatorial(ra_before, dec_before, NSIDE)
    m_after_equ = build_healpix_map_equatorial(ra_after, dec_after, NSIDE)

    plot_healpix_density(
        m_before_equ,
        title=f"DESI Y3 quasars (z={zmin}–{zmax}) before footprint cut (equatorial, NSIDE={NSIDE})",
        outfile=OUTDIR / f"density_before_footprint_nside{NSIDE}_equ.png",
    )

    plot_healpix_density(
        m_after_equ,
        title=f"DESI Y3 quasars (z={zmin}–{zmax}) after footprint cut (equatorial, NSIDE={NSIDE})",
        outfile=OUTDIR / f"density_after_footprint_nside{NSIDE}_equ.png",
    )

    # --- HEALPix maps: Galactic ---
    m_before_gal = build_healpix_map_galactic(ra_before, dec_before, NSIDE)
    m_after_gal = build_healpix_map_galactic(ra_after, dec_after, NSIDE)

    plot_healpix_density(
        m_before_gal,
        title=f"DESI Y3 quasars (z={zmin}–{zmax}) before footprint cut (Galactic, NSIDE={NSIDE})",
        outfile=OUTDIR / f"density_before_footprint_nside{NSIDE}_gal.png",
    )

    plot_healpix_density(
        m_after_gal,
        title=f"DESI Y3 quasars (z={zmin}–{zmax}) after footprint cut (Galactic, NSIDE={NSIDE})",
        outfile=OUTDIR / f"density_after_footprint_nside{NSIDE}_gal.png",
    )


if __name__ == "__main__":
    main()
