#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(Path.home()/ "project/results/mcmc/per_bin"))
    ap.add_argument("--out-fig-dir", default=str(Path.home()/ "project/results/figures/mcmc"))
    ap.add_argument("--out-csv", default=str(Path.home()/ "project/results/tables/mcmc_param_evolution.csv"))
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--color-chains", action="store_true")
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--style", default=None, help="matplotlib style name, e.g. seaborn-v0_8-colorblind")
    ap.add_argument("--palette", default="tab10", help="matplotlib colormap / palette name")
    ap.add_argument("--chain-colors", default="", help="CSV hex list for chains, e.g. #1f77b4,#ff7f0e,...")
    return ap.parse_args()

def chain_colors(n, palette, chain_hex):
    if chain_hex:
        cols = [c.strip() for c in chain_hex.split(",") if c.strip()]
        if len(cols) >= n:
            return cols[:n]
    cmap = plt.get_cmap(palette)
    if hasattr(cmap, "colors") and len(cmap.colors) >= n:
        return [cmap.colors[i] for i in range(n)]
    if n == 1:
        return [cmap(0.5)]
    return [cmap(i/(n-1)) for i in range(n)]

def load_npz(fpath):
    arr = np.load(fpath)
    keys = set(arr.keys())
    # prefer base-10 logs if present; else convert natural logs
    def pick(name10, namee):
        if name10 in keys:
            return np.asarray(arr[name10]), True
        elif namee in keys:
            return np.asarray(arr[namee]) / np.log(10.0), True
        else:
            return None, False
    m0   = np.asarray(arr["m0"])        if "m0" in keys else None
    b    = np.asarray(arr["b"])         if "b"  in keys else None
    alpha= np.asarray(arr["alpha"])     if "alpha" in keys else None
    beta = np.asarray(arr["beta"])      if "beta"  in keys else None
    log10_Nphi, _  = pick("log10_Nphi", "log_Nphi")
    log10_Lstar,_  = pick("log10_Lstar","log_Lstar")
    zmin = float(arr["zmin"]) if "zmin" in keys else np.nan
    zmax = float(arr["zmax"]) if "zmax" in keys else np.nan
    return {
      "m0": m0, "b": b, "alpha": alpha, "beta": beta,
      "log10_Nphi": log10_Nphi, "log10_Lstar": log10_Lstar,
      "zmin": zmin, "zmax": zmax
    }

def split_by_chain(vec, n_chains):
    if vec is None: return None
    n = vec.shape[0]
    if n_chains <= 1: return [vec]
    per = n // n_chains
    return [vec[i*per:(i+1)*per] for i in range(n_chains)]

def make_corner_per_file(fpath, outdir, chains, color_chains, palette, chain_hex, thin):
    data = load_npz(fpath)
    # select parameters that exist
    order = ["m0","b","alpha","beta","log10_Nphi","log10_Lstar"]
    labels= [r"$m_0$", r"$b$", r"$\alpha$", r"$\beta$", r"$\log_{10}N_\phi$", r"$\log_{10}L_\star$"]
    keep_idx, vals = [], []
    for i, p in enumerate(order):
        v = data[p]
        if v is not None:
            vals.append(v[::thin])
            keep_idx.append(i)
    if len(vals) == 0:
        print(f"[skip] no parameters found in {fpath.name}")
        return
    vals = np.vstack(vals).T  # (Nsamp, Nparam_kept)
    kept_labels = [labels[i] for i in keep_idx]

    # chain coloring
    fig = None
    if color_chains and chains > 1:
        cols = chain_colors(chains, palette, chain_hex)
        # split each param vector by chain and stack back per chain
        split_cols = [split_by_chain(load_npz(fpath)[p], chains) for p in order if load_npz(fpath)[p] is not None]
        per_chain_samples = []
        for c in range(chains):
            pcs = [pc[c][::thin] for pc in split_cols]  # list of arrays for this chain
            per_chain_samples.append(np.vstack(pcs).T)
        # plot first chain
        fig = corner.corner(per_chain_samples[0], labels=kept_labels, color=cols[0],
                            show_titles=True, title_fmt=".2f", quantiles=[0.16,0.5,0.84])
        # overlay others
        for ci in range(1, chains):
            corner.corner(per_chain_samples[ci], color=cols[ci], fig=fig,
                          labels=kept_labels, show_titles=False)
    else:
        fig = corner.corner(vals, labels=kept_labels, show_titles=True,
                            title_fmt=".2f", quantiles=[0.16,0.5,0.84])

    out = outdir / f"corner_{fpath.stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[corner] {out}")

def summarize_file(fpath, thin):
    d = load_npz(fpath)
    zc = 0.5*(d["zmin"]+d["zmax"])
    row = {"file": fpath.name, "z": zc, "zmin": d["zmin"], "zmax": d["zmax"]}
    for p in ["m0","b","alpha","beta","log10_Nphi","log10_Lstar"]:
        v = d[p]
        if v is None: continue
        v = v[::thin]
        q16, q50, q84 = np.quantile(v, [0.16,0.5,0.84])
        row[f"{p}_16"] = q16; row[f"{p}_50"] = q50; row[f"{p}_84"] = q84
    return row

def plot_evolution(df, outdir, style, palette):
    params = ["m0","b","alpha","beta","log10_Nphi","log10_Lstar"]
    names  = [r"$m_0$", r"$b$", r"$\alpha$", r"$\beta$", r"$\log_{10}N_\phi$", r"$\log_{10}L_\star$"]
    cmap = plt.get_cmap(palette)
    for i, p in enumerate(params):
        if f"{p}_50" not in df.columns: continue
        y  = df[f"{p}_50"].to_numpy()
        ylo= df[f"{p}_50"].to_numpy() - df[f"{p}_16"].to_numpy()
        yhi= df[f"{p}_84"].to_numpy() - df[f"{p}_50"].to_numpy()
        z  = df["z"].to_numpy()
        order = np.argsort(z)
        plt.figure(figsize=(6.0,4.0))
        plt.errorbar(z[order], y[order], yerr=[ylo[order], yhi[order]], fmt="o-", capsize=3)
        plt.xlabel("Redshift z")
        plt.ylabel(names[i])
        plt.tight_layout()
        out = outdir / f"evolution_{p}.png"
        plt.savefig(out, dpi=220)
        plt.close()
        print(f"[evolution] {out}")

def main():
    args = parse_args()
    in_dir  = Path(args.input_dir)
    out_dir = Path(args.out_fig_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv);    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.style:
        try:
            plt.style.use(args.style)
        except Exception as e:
            print(f"[warn] style {args.style} not found ({e}); continuing with default.")

    # find bins
    pat = re.compile(r"y3_bin(\d+)_z([0-9.]+)-([0-9.]+)\.npz$")
    files = []
    for p in sorted(in_dir.glob("y3_bin*_z*.npz")):
        if pat.match(p.name):
            files.append(p)
    if not files:
        print(f"[err] no npz files in {in_dir}")
        return

    # corners
    for p in files:
        make_corner_per_file(
            p, out_dir, chains=args.chains, color_chains=args.color_chains,
            palette=args.palette, chain_hex=args.chain_colors, thin=args.thin
        )

    # summaries + evolution
    rows = [summarize_file(p, thin=args.thin) for p in files]
    df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[table] {out_csv}")

    plot_evolution(df, out_dir, args.style, args.palette)

if __name__ == "__main__":
    main()