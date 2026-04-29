#!/usr/bin/env python3
# Per-bin NumPyro MCMC for DESI Y3 QLF
# all-case only, order=1 sigma_m(m), per-object mu

import os
import sys
from pathlib import Path
import argparse
import numpy as np

from jax import random, config as jax_config
jax_config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer.initialization import init_to_uniform

# project import
ROOT = Path.home() / "project"
sys.path.insert(0, str(ROOT / "src"))

from y3pipe.qlf import ln_likelihood_bin

# ----------------------------
# Paths
# ----------------------------
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "results" / "mcmc" / "per_bin" / "order1_all"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# z bins
# ----------------------------
ZBINS = {
    1: (2.3, 2.4),
    2: (2.4, 2.5),
    3: (2.5, 2.6),
    4: (2.6, 2.8),
}

# ----------------------------
# Priors
# Use the narrowed priors that worked in notebook tests
# ----------------------------
def natlog_bounds():
    logN_lo, logN_hi = 6.0, 11.0
    logL_lo, logL_hi = 23.0, 26.0
    return logN_lo, logN_hi, logL_lo, logL_hi


# ----------------------------
# Load per-bin arrays
# ----------------------------
def load_bin_arrays(idx: int):
    zmin, zmax = ZBINS[idx]
    npz = DATA / f"y3_bin{idx}_z{zmin}-{zmax}_arrays.npz"
    d = np.load(npz)

    m       = d["apr_mag_r"].astype(np.float64)
    sigma_m = d["sigma_m"].astype(np.float64)
    k       = d["k"].astype(np.float64)
    sigma_k = d["sigma_k"].astype(np.float64)
    mu      = d["mu"].astype(np.float64)

    return m, sigma_m, k, sigma_k, mu, zmin, zmax, npz


# ----------------------------
# Fit order-1 sigma_m(m)
# log10 sigma_m = a1 * (m - m_ref) + a0
# ----------------------------
def fit_sigma_order1(m, sigma_m, m_ref=22.0, nbins=20, min_per_bin=20):
    m = np.asarray(m, dtype=float)
    sigma_m = np.asarray(sigma_m, dtype=float)
    y = np.log10(sigma_m)

    edges = np.linspace(m.min(), m.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    xm, ym, wn = [], [], []
    for i in range(nbins):
        sel = (m >= edges[i]) & (m < edges[i + 1])
        n = int(sel.sum())
        if n < min_per_bin:
            continue
        xm.append(centers[i])
        ym.append(np.median(y[sel]))
        wn.append(np.sqrt(n))

    xm = np.asarray(xm)
    ym = np.asarray(ym)
    wn = np.asarray(wn)

    if len(xm) < 3:
        raise RuntimeError(
            f"Not enough populated bins to fit order-1 sigma model: used={len(xm)}"
        )

    coeff = np.polyfit(xm - m_ref, ym, deg=1, w=wn)

    fit_info = {
        "m_ref": float(m_ref),
        "nbins": int(nbins),
        "min_per_bin": int(min_per_bin),
        "m_centers": xm,
        "y_median": ym,
        "weights": wn,
        "nbins_used": int(len(xm)),
    }
    return coeff, fit_info


# ----------------------------
# NumPyro model
# ----------------------------
def make_model(like, m_j):
    logN_lo, logN_hi, logL_lo, logL_hi = natlog_bounds()

    def model():
        # selection params
        m0 = numpyro.sample("m0", dist.Uniform(20.0, 24.0))
        b  = numpyro.sample("b",  dist.Uniform(1.0, 3.0))

        # QLF params
        log_Nphi  = numpyro.sample("log_Nphi",  dist.Uniform(logN_lo, logN_hi))
        log_Lstar = numpyro.sample("log_Lstar", dist.Uniform(logL_lo, logL_hi))
        beta      = numpyro.sample("beta",      dist.Uniform(-7.0, -2.0))
        gap       = numpyro.sample("gap",       dist.Uniform(1e-3, 5.0))
        alpha     = numpyro.deterministic("alpha", beta + gap)

        logL = like(m0, b, m_j, log_Nphi, log_Lstar, alpha, beta)
        numpyro.factor("log_likelihood", logL)

    return model


# ----------------------------
# Save helper
# ----------------------------
def _resolve_and_save_npz(outfile: Path, **arrays):
    real_path = Path(os.path.realpath(str(outfile)))
    real_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(real_path, **arrays)
    print(f"[save] wrote {real_path} (resolved from {outfile})")


# ----------------------------
# Main
# ----------------------------
def main():
    print("JAX backend:", jax.default_backend())
    print("Devices:", jax.devices())

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-index", type=int, help="1..4 (if omitted, uses SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--num-warmup", type=int, default=10000)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--target-accept", type=float, default=0.9)

    # sigma-fit controls
    ap.add_argument("--m-ref", type=float, default=22.0)
    ap.add_argument("--sigma-nbins", type=int, default=20)
    ap.add_argument("--sigma-min-per-bin", type=int, default=20)

    args = ap.parse_args()

    # resolve bin index
    idx = args.bin_index
    if idx is None:
        tid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if tid:
            idx = int(tid)
    if idx not in ZBINS:
        raise SystemExit(f"bin-index must be one of {sorted(ZBINS.keys())}; got {idx}")

    # load full bin (all-case only)
    m, sigma_m, k, sigma_k, mu, zmin, zmax, in_npz = load_bin_arrays(idx)
    print(f"[bin {idx}] z={zmin}-{zmax} N={m.size:,} from: {in_npz.name}")

    # fit sigma(m) model for this full bin
    coeff, fit_info = fit_sigma_order1(
        m, sigma_m,
        m_ref=args.m_ref,
        nbins=args.sigma_nbins,
        min_per_bin=args.sigma_min_per_bin,
    )
    print(f"[bin {idx}] order-1 sigma coeff = {coeff}")

    # cast arrays
    m_j       = jnp.asarray(m)
    sigma_m_j = jnp.asarray(sigma_m)
    k_j       = jnp.asarray(k)
    sigma_k_j = jnp.asarray(sigma_k)
    mu_j      = jnp.asarray(mu)

    # build likelihood
    like = ln_likelihood_bin(
        sigma_m=sigma_m_j,
        k=k_j,
        sigma_k=sigma_k_j,
        zmin=zmin,
        zmax=zmax,
        order=1,
        sigma_coeff=coeff,
        mu=None,            # use constant mu(z_center) inside the likelihood
        m_ref=args.m_ref,
    )

    model = make_model(like, m_j)

    # avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    nuts = NUTS(
        model,
        target_accept_prob=args.target_accept,
        init_strategy=init_to_uniform(radius=0.3),
    )

    base_key = random.PRNGKey(args.seed + idx)
    chain_keys = random.split(base_key, args.num_chains)

    mcmc = MCMC(
        nuts,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="sequential",
        progress_bar=True,
    )

    print(
        f"[bin {idx}] starting MCMC: {args.num_chains} chains × "
        f"{args.num_samples} samples (warmup {args.num_warmup})"
    )

    mcmc.run(chain_keys)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # save output
    outfile = OUT / f"y3_bin{idx}_z{zmin}-{zmax}_all_order1_w{args.num_warmup}_s{args.num_samples}_c{args.num_chains}.npz"
    _resolve_and_save_npz(
        outfile,
        **{kname: np.asarray(v) for kname, v in samples.items()},
        bin_index=int(idx),
        case="all",
        order=1,
        zmin=float(zmin),
        zmax=float(zmax),
        N=int(m.size),
        sigma_coeff=np.asarray(coeff),
        m_ref=float(args.m_ref),
        sigma_fit_nbins=int(args.sigma_nbins),
        sigma_fit_min_per_bin=int(args.sigma_min_per_bin),
        input_arrays=str(in_npz),
    )

    print(f"[bin {idx}] done.")


if __name__ == "__main__":
    main()