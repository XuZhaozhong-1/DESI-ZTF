#!/usr/bin/env python3
# Per-bin NumPyro MCMC for DESI Y3 QLF (natural-log params), with split RNG keys
# and safe overwrite of symlink targets.

import re
from pathlib import Path
import os
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

# Your likelihood: ln_likelihood_bin(zmin, zmax)(...)
from y3pipe.qlf import ln_likelihood_bin

ROOT    = Path.home() / "project"
ARR_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "mcmc_test" / "per_bin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LN10 = float(np.log(10.0))

# -------- discover per-bin arrays --------
pat = re.compile(r"y3_bin(\d+)_z([0-9.]+)-([0-9.]+)_arrays\.npz$")
bins = []
for p in sorted(ARR_DIR.glob("y3_bin*_arrays.npz")):
    m = pat.match(p.name)
    if not m:
        continue
    idx  = int(m.group(1))
    zmin = float(m.group(2))
    zmax = float(m.group(3))
    bins.append(dict(idx=idx, zmin=zmin, zmax=zmax, file=p))

if not bins:
    raise SystemExit(f"No per-bin arrays found in {ARR_DIR}")

print("Discovered bins:")
for b in bins:
    print(f"  bin{b['idx']}: z={b['zmin']}-{b['zmax']}  {Path(b['file']).name}")

# -------- one-bin runner --------
def run_one_bin(bininfo, warmup=1000, n_samples=1500, seed=1234, n_chains=4):
    # keep BLAS/OpenMP from over-spawning threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    p = Path(bininfo["file"])
    arr = np.load(p)

    # required arrays
    m       = jnp.asarray(arr["apr_mag_r"])
    sigma_m = jnp.asarray(arr["sigma_m"])
    k       = jnp.asarray(arr["k"])
    sigma_k = jnp.asarray(arr["sigma_k"])
    mu      = jnp.asarray(arr["mu"])

    # z-range (prefer values embedded in file)
    zmin = float(arr["zmin"][0]) if "zmin" in arr else float(bininfo["zmin"])
    zmax = float(arr["zmax"][0]) if "zmax" in arr else float(bininfo["zmax"])

    pretty = p.stem  # e.g., y3_bin1_z2.3-2.4_arrays
    print(f"\nRunning bin: {pretty}  ({p.name})  z={zmin}-{zmax}  N={m.size}")
    print("JAX backend:", jax.default_backend())

    like = ln_likelihood_bin(zmin, zmax)

    def model():
        # selection params
        m0 = numpyro.sample("m0", dist.Uniform(15.0, 25.0))
        b  = numpyro.sample("b",  dist.Uniform(1.0,  3.0))

        # sample in base-10, convert to natural log for the likelihood
        log10_Nphi  = numpyro.sample("log10_Nphi",  dist.Uniform(0.0,  10.0))
        log10_Lstar = numpyro.sample("log10_Lstar", dist.Uniform(40.0, 50.0))
        log_Nphi    = numpyro.deterministic("log_Nphi",  log10_Nphi  * LN10)
        log_Lstar   = numpyro.deterministic("log_Lstar", log10_Lstar * LN10)

        # double power-law with alpha > beta
        beta = numpyro.sample("beta", dist.Uniform(-7.0, -2.0))
        gap  = numpyro.sample("gap",  dist.Uniform(1e-3, 5.0))
        alpha = numpyro.deterministic("alpha", beta + gap)

        ll = like(m0, b, m, sigma_m, k, sigma_k, mu, log_Nphi, log_Lstar, alpha, beta)
        numpyro.factor("log_likelihood", ll)

    kernel = NUTS(model, target_accept_prob=0.85)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=True,
    )

    # unique RNG per chain to avoid correlated initializations
    base_key = jax.random.PRNGKey(seed + bininfo["idx"])
    chain_keys = jax.random.split(base_key, n_chains)

    mcmc.run(chain_keys)  # pass split keys so each chain is distinct
    #mcmc.run(base_key)
    mcmc.print_summary()

    # save posterior samples; write to the symlink target if OUT_DIR/filename is a symlink
    post = mcmc.get_samples()
    base_name = pretty.replace("_arrays", "") + ".npz"  # y3_binX_zA-B.npz
    out_path  = OUT_DIR / base_name

    # If OUT_DIR/base_name is a symlink, resolve to the true file and overwrite that.
    # Otherwise, write to OUT_DIR/base_name as-is.
    target_path = out_path.resolve() if out_path.is_symlink() else out_path

    np.savez_compressed(
        target_path,
        **{k: np.asarray(v) for k, v in post.items()},
        zmin=zmin, zmax=zmax
    )
    print(f"Saved samples -> {target_path}")

def main():
    for b in bins:
        # tune these as you like:
        run_one_bin(b, warmup=5000, n_samples=5000, seed=1234, n_chains=2)

if __name__ == "__main__":
    main()
