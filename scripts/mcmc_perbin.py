#!/usr/bin/env python3
# Per-bin NumPyro MCMC for DESI Y3 QLF (natural-log params), publication-grade version
# Includes fully independent RNG keys and randomized initialization per chain.

import os
from pathlib import Path
import argparse
import numpy as np

# --- JAX / NumPyro setup ---
from jax import random, config as jax_config
jax_config.update("jax_enable_x64", True)     # double precision
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer.initialization import init_to_uniform

# --- Likelihood function ---
from y3pipe.qlf import ln_likelihood_bin

ROOT = Path.home() / "project"

# ---- z-bins (index, zmin, zmax) ----
ZBINS = {
    1: (2.3, 2.4),
    2: (2.4, 2.5),
    3: (2.5, 2.6),
    4: (2.6, 2.8),
}

def natlog_bounds():
    ln10 = np.log(10.0)
    # log10_Nphi ∈ [0, 10]      -> log_Nphi ∈ [0*ln10, 10*ln10]
    # log10_Lstar ∈ [44, 47.5]  -> log_Lstar ∈ [44*ln10, 47.5*ln10]
    return (0.0*ln10, 10.0*ln10, 44.0*ln10, 47.5*ln10)

def load_bin_arrays(idx: int):
    """
    Load per-bin arrays from:
      data/processed/y3_bin{idx}_z{zmin}-{zmax}_arrays.npz
    Required keys: apr_mag_r, sigma_m, k, sigma_k, mu
    """
    zmin, zmax = ZBINS[idx]
    npz = ROOT / f"data/processed/y3_bin{idx}_z{zmin}-{zmax}_arrays.npz"
    d = np.load(npz)
    m        = jnp.asarray(d["apr_mag_r"])
    sigma_m  = jnp.asarray(d["sigma_m"])
    k        = jnp.asarray(d["k"])
    sigma_k  = jnp.asarray(d["sigma_k"])
    mu       = jnp.asarray(d["mu"])
    return m, sigma_m, k, sigma_k, mu, zmin, zmax, npz

def make_model(like, m, sigma_m, k, sigma_k, mu):
    logN_lo, logN_hi, logL_lo, logL_hi = natlog_bounds()

    def model():
        # Selection (per-bin)
        m0 = numpyro.sample("m0", dist.Uniform(15.0, 25.0))
        b  = numpyro.sample("b",  dist.Uniform(1.0, 3.0))

        # QLF parameters (natural logs)
        log_Nphi  = numpyro.sample("log_Nphi",  dist.Uniform(logN_lo,  logN_hi))
        log_Lstar = numpyro.sample("log_Lstar", dist.Uniform(logL_lo, logL_hi))
        beta      = numpyro.sample("beta",      dist.Uniform(-7.0, -2.0))
        gap       = numpyro.sample("gap",       dist.Uniform(1e-3, 5.0))
        alpha     = numpyro.deterministic("alpha", beta + gap)  # enforce alpha > beta

        logL = like(m0, b, m, sigma_m, k, sigma_k, mu,
                    log_Nphi, log_Lstar, alpha, beta)
        numpyro.factor("log_likelihood", logL)

    return model

def _resolve_and_save_npz(outfile: Path, **arrays):
    """
    Overwrite the real target even if 'outfile' is a symlink.
    """
    real_path = Path(os.path.realpath(str(outfile)))
    real_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(real_path, **arrays)
    print(f"[save] wrote {real_path} (resolved from {outfile})")

def main():
    print("JAX backend:", jax.default_backend())

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-index", type=int, help="1..4 (if omitted, uses SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--num-warmup", type=int, default=10000)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    # Resolve bin index from CLI or Slurm array
    idx = args.bin_index
    if idx is None:
        tid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if tid:
            idx = int(tid)
    if idx not in ZBINS:
        raise SystemExit(f"bin-index must be 1..4; got {idx}")

    # Load data for this bin
    m, sigma_m, k, sigma_k, mu, zmin, zmax, in_npz = load_bin_arrays(idx)
    print(f"[bin {idx}] z={zmin}-{zmax}  N={m.size}  from: {in_npz.name}")

    # Build likelihood + model
    like  = ln_likelihood_bin(zmin, zmax)
    model = make_model(like, m, sigma_m, k, sigma_k, mu)

    # Avoid thread oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # --- NUTS Sampler with independent chain initialization ---
    nuts = NUTS(
        model,
        target_accept_prob=0.9,
        init_strategy=init_to_uniform(radius=0.5),
    )

    # --- Independent RNG keys per chain ---
    base_key = random.PRNGKey(args.seed + idx)
    chain_keys = random.split(base_key, args.num_chains)

    mcmc = MCMC(
        nuts,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="sequential",   # safer for GPU nodes
        progress_bar=True,
    )

    print(f"[bin {idx}] starting MCMC: {args.num_chains} chains × "
          f"{args.num_samples} samples (warmup {args.num_warmup})")

    mcmc.run(chain_keys)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Save posterior samples (resolve symlink target if present)
    outdir = ROOT / "results/mcmc/per_bin"
    outfile = outdir / f"y3_bin{idx}_z{zmin}-{zmax}.npz"
    _resolve_and_save_npz(
        outfile,
        **{k: np.asarray(v) for k, v in samples.items()},
        zmin=zmin, zmax=zmax,
    )
    print(f"[bin {idx}] done.")

if __name__ == "__main__":
    main()
