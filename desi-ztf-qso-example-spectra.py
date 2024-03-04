#!/usr/bin/env python

import os
from glob import glob
from time import time
import numpy as np
import fitsio
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--tid",
        help="TARGETID to plot (default=None)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--outpng",
        help="outpng png file (default=None)",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    for kwargs in args._get_kwargs():
        print(kwargs)
    return args


# get the desi wavelengths (same for all files)
def get_ws(ardir):
    fn = sorted(
        glob(os.path.join(ardir, "pernight-spectra", "desi-ztf-qso-iron-*-*.fits"))
    )[0]
    ws = fitsio.read(fn, "BRZ_WAVE")
    return ws


# get the {TILEID,NIGHT} sets where the TARGETID has been observed
def get_tid_tileids_nights(tid, all_tids, all_tileids, all_nights):
    sel = all_tids == tid
    return all_tileids[sel], all_nights[sel]


# read the spectra
def get_indiv_spectra(tid, tileids, nights, ardir, nwave):

    nobs = len(tileids)

    # read the spectra
    fs = np.zeros((nobs, nwave))  # flux
    ivs = np.zeros((nobs, nwave))  # inverse variance

    # loop on (tileids, nights)
    for i, (tileid, night) in enumerate(zip(tileids, nights)):

        fn = os.path.join(
            ardir,
            "pernight-spectra",
            "desi-ztf-qso-iron-{}-{}.fits".format(tileid, night),
        )

        # first get the row corresponding to TARGETID
        # (for a given TILEID, a TARGETID can appear only once max.)
        tmp_fm = fitsio.read(fn, "FIBERMAP", columns=["TARGETID"])
        tmp_i = np.where(tmp_fm["TARGETID"] == tid)[0][0]

        # now read the flux, ivar only for that row

        # this is more intuitive, but it loads the whole image first,
        #   then extracts the relevant row, so it s ~slow
        # fs[i, :] = fitsio.read(fn, "BRZ_FLUX")[tmp_i, :]
        # ivs[i, :] = fitsio.read(fn, "BRZ_IVAR")[tmp_i, :]

        # this way is *much* faster, with reading only the relevant row
        #   note: fitsio wants/works with a slice, not an integer...
        h = fitsio.FITS(fn)
        tmp_slice = slice(tmp_i, tmp_i + 1, 1)
        fs[i, :] = h["BRZ_FLUX"][tmp_slice, :]
        ivs[i, :] = h["BRZ_IVAR"][tmp_slice, :]

    return fs, ivs


# handle nan s for smoothing..
# https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
def get_smooth(fs, ivs, gauss_smooth):
    tmp0fs = fs.copy()
    tmp0fs[ivs == 0] = 0
    tmp1fs = 1 + 0 * fs.copy()
    tmp1fs[ivs == 0] = 0
    tmp0smfs = gaussian_filter1d(tmp0fs, gauss_smooth, mode="constant", cval=0)
    tmp1smfs = gaussian_filter1d(tmp1fs, gauss_smooth, mode="constant", cval=0)
    return tmp0smfs / tmp1smfs


def plot_tid_spectra(outpng, tid, ws, fs, ivs):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(fs.shape[0]):
        smfs_i = get_smooth(fs[i], ivs[i], 5)
        ax.plot(ws, smfs_i, lw=0.5)
    ax.set_title("TARGETID = {}".format(tid))
    ax.set_xlabel("Observed wavelength [A]")
    ax.set_ylabel("Flux [erg / cm2 / s / A]")
    ax.grid()
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def main():

    args = parse()

    # Anand s folder
    ardir = os.path.join(os.getenv("PSCRATCH"), "desi-ztf-qso")
    # summary table
    sumfn = os.path.join(ardir, "desi-ztf-qso-iron-pernight-summary.fits")

    # wavelengths
    ws = get_ws(ardir)
    nwave = len(ws)

    # read the master table
    start = time()
    d = fitsio.read(sumfn, "FIBERMAP")
    print("reading {} done (took {:.1f}s)".format(sumfn, time() - start))

    # get the (TILEID,NIGHT) sets for that TARGETID
    tileids, nights = get_tid_tileids_nights(
        args.tid, d["TARGETID"], d["TILEID"], d["NIGHT"]
    )

    # get fluxes and ivars
    start = time()
    fs, ivs = get_indiv_spectra(args.tid, tileids, nights, ardir, nwave)
    print(
        "done extracting {} spectra for TARGETID={} (took {:.1f}s)".format(
            len(tileids), args.tid, time() - start
        )
    )

    # plot
    plot_tid_spectra(args.outpng, args.tid, ws, fs, ivs)
    print("done plotting {}".format(args.outpng))


if __name__ == "__main__":
    main()
