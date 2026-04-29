#!/usr/bin/env python3
"""
Run DESI Y3 per-bin MCMC for one "case": all / upper stripe / lower stripe.

Behavior:
  - Load processed arrays for a bin:
      data/processed/y3_bin{idx}_z{zmin}-{zmax}_arrays.npz
    keys: apr_mag_r, sigma_m, k, sigma_k, mu  (mu array is ignored for likelihood)

  - Load precomputed stripe masks:
      data/processed/stripes/bin{idx}_z{zmin}-{zmax}_stripes.npz
    keys: upper, lower  (valid may exist, but we don't rely on it)

  - Load prefit sigma(m) model for this bin+case:
      results/sigma_m_models/sigma_m_model_bin{idx}_z{zmin}-{zmax}_{case}.npz
    keys: coeff, m_ref, deg, (plus diagnostics)

  - Likelihood uses sigma(m) internally (from src/y3pipe/qlf.py):
      ln_likelihood_bin(zmin, zmax, sigma_coeff=coeff, m_ref=m_ref)

  - Use mu(z_center) as a scalar (broadcasted)

Outputs:
  results/mcmc/stripe_cases/
      bin{idx}_z{zmin}-{zmax}_{case}_w{W}_s{S}_c{C}.npz

Example:
  python scripts/mcmc_perbin_case.py --bin-index 2 --case upper \
      --num-warmup 10000 --num-samples 10000 --num-chains 4
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np

# ----------------------------
# Paths / import project
# ----------------------------
ROOT = Path.home() / "project"
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "processed"
STRP = DATA / "stripes"
SIGM = ROOT / "results" / "sigma_m_models"
OUTM = ROOT / "results" / "mcmc" / "stripe_cases"
OUTM.mkdir(parents=True, exist_ok=True)

# ----------------------------
# JAX / NumPyro
# ----------------------------
import jax
import jax.numpy as jnp
from jax import random
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer.initialization import init_to_uniform

from astropy.cosmology import Planck18

# Your likelihood (now sigma(m)-aware)
from y3pipe.qlf import ln_likelihood_bin


# ----------------------------
# Bins
# ----------------------------
ZBINS = {
    1: (2.3, 2.4),
    2: (2.4, 2.5),
    3: (2.5, 2.6),
    4: (2.6, 2.7),
    5: (2.7, 2.8),
}


def natlog_bounds():
    ln10 = np.log(10.0)
    return (0.0 * ln10, 10.0 * ln10, 44.0 * ln10, 47.5 * ln10)


def load_bin_arrays(idx: int, zmin: float, zmax: float):
    """Load processed arrays for a bin."""
    npz = DATA / f"y3_bin{idx}_z{zmin}-{zmax}_arrays.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing processed arrays: {npz}")
    d = np.load(npz)
    m = d["apr_mag_r"].astype(np.float64)
    sigma_m = d["sigma_m"].astype(np.float64)     # used only for masking/diagnostics
    k = d["k"].astype(np.float64)
    sigma_k = d["sigma_k"].astype(np.float64)
    mu_arr = d["mu"].astype(np.float64)           # NOT used in likelihood; only for optional z-subcut
    return m, sigma_m, k, sigma_k, mu_arr, npz


def load_stripe_masks(idx: int, zmin: float, zmax: float):
    """Load precomputed stripe masks for a bin."""
    tag = f"bin{idx}_z{zmin}-{zmax}"
    npz = STRP / f"{tag}_stripes.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing stripe masks: {npz}")
    d = np.load(npz)
    upper = d["upper"].astype(bool)
    lower = d["lower"].astype(bool)
    return upper, lower, npz


def load_sigma_model(idx: int, zmin: float, zmax: float, case: str):
    """
    Load sigma(m) polynomial coeff for given bin+case:
    results/sigma_m_models/sigma_m_model_bin{idx}_z{zmin}-{zmax}_{case}.npz

    """
    tag = f"bin{idx}_z{zmin}-{zmax}"
    f = SIGM / f"sigma_m_model_{tag}_{case}.npz"
    d = np.load(f, allow_pickle=True)
    coeff = np.asarray(d["coeff"], dtype=float)
    m_ref = float(d["m_ref"]) if "m_ref" in d else 22.0
    deg = int(d["deg"]) if "deg" in d else (len(coeff) - 1)
    return coeff, m_ref, deg, f

def make_numpyro_model(like, m_j, k_j, sigma_k_j, mu_scalar):
    """NumPyro model; mu is a scalar broadcasted inside likelihood."""
    def model():
        logN_lo, logN_hi, logL_lo, logL_hi = natlog_bounds()

        # selection params
        m0 = numpyro.sample("m0", dist.Uniform(15.0, 25.0))
        b = numpyro.sample("b", dist.Uniform(1.0, 3.0))

        # QLF params
        log_Nphi = numpyro.sample("log_Nphi", dist.Uniform(logN_lo, logN_hi))
        log_Lstar = numpyro.sample("log_Lstar", dist.Uniform(logL_lo, logL_hi))
        beta = numpyro.sample("beta", dist.Uniform(-7.0, -2.0))
        gap = numpyro.sample("gap", dist.Uniform(1e-3, 5.0))
        alpha = numpyro.deterministic("alpha", beta + gap)

        logL = like(m0, b, m_j, k_j, sigma_k_j, mu_scalar,
                    log_Nphi, log_Lstar, alpha, beta)
        numpyro.factor("log_likelihood", logL)

    return model


def run_mcmc(model, seed, num_warmup, num_samples, num_chains, target_accept=0.9):
    nuts = NUTS(model, target_accept_prob=target_accept, init_strategy=init_to_uniform(radius=0.5))
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=True,
    )
    keys = random.split(random.PRNGKey(seed), num_chains)
    mcmc.run(keys)
    return mcmc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-index", type=int, required=True, choices=sorted(ZBINS.keys()))
    ap.add_argument("--case", type=str, required=True, choices=["all", "upper", "lower"])
    ap.add_argument("--num-warmup", type=int, default=10000)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--target-accept", type=float, default=0.9)

    args = ap.parse_args()

    print("JAX backend:", jax.default_backend())
    print("NumPyro:", numpyro.__version__)

    idx = args.bin_index
    zmin, zmax = ZBINS[idx]
    tag = f"bin{idx}_z{zmin}-{zmax}"

    # ---------- load data ----------
    m, sigma_m, k, sigma_k, mu_arr, in_npz = load_bin_arrays(idx, zmin, zmax)
    print(f"[load] {tag} N={len(m):,} from {in_npz}")

    # ---------- load masks ----------
    upper, lower, mask_npz = load_stripe_masks(idx, zmin, zmax)
    print(f"[masks] upper={upper.sum():,} lower={lower.sum():,} from {mask_npz.name}")

    # ---------- choose case ----------
    if args.case == "all":
        use = np.ones_like(m, dtype=bool)
    elif args.case == "upper":
        use = upper.copy()
    else:
        use = lower.copy()


    # ---------- load sigma(m) coeff for this case ----------
    coeff, m_ref, deg, sigma_file = load_sigma_model(idx, zmin, zmax, args.case)
    print(f"[sigma] loaded coeff deg={deg} m_ref={m_ref} from {sigma_file.name}")

    # ---------- mu(z_center) scalar ----------
    z_center = 0.5 * (zmin + zmax)
    mu_cen = float(Planck18.distmod(z_center).value)
    mu_scalar = jnp.asarray(mu_cen)
    print(f"[mu] using mu(z_center={z_center:.3f}) = {mu_cen:.6f} (scalar)")

    # ---------- likelihood ----------
    like = ln_likelihood_bin(zmin, zmax, sigma_coeff=coeff, m_ref=m_ref)

    # ---------- build MCMC arrays ----------
    m_j = jnp.asarray(m[use])
    k_j = jnp.asarray(k[use])
    sigma_k_j = jnp.asarray(sigma_k[use])

    model = make_numpyro_model(like, m_j, k_j, sigma_k_j, mu_scalar)

    # seed includes bin+case
    seed_case = {"all": 0, "upper": 11, "lower": 22}[args.case]
    seed = args.seed + 1000 * idx + seed_case
    print(f"[mcmc] warmup={args.num_warmup} samples={args.num_samples} chains={args.num_chains} seed={seed}")

    mcmc = run_mcmc(
        model,
        seed=seed,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept=args.target_accept,
    )

    mcmc.print_summary()
    samples = mcmc.get_samples()

    # ---------- save ----------
    out_tag = tag
    if ztag:
        out_tag += f"_{ztag}"

    outfile = OUTM / f"{out_tag}_{args.case}_w{args.num_warmup}_s{args.num_samples}_c{args.num_chains}.npz"

    np.savez(
        outfile,
        **{k: np.asarray(v) for k, v in samples.items()},
        sigma_model_file=str(sigma_file),
        sigma_coeff=np.asarray(coeff),
        m_ref=float(m_ref),
        case=args.case,
        bin_index=int(idx),
        zmin=float(zmin),
        zmax=float(zmax),
        z_center=float(z_center),
        mu_center=float(mu_cen),
        N_case=int(use.sum()),
        N_mcmc=int(m_j.size),
        input_arrays=str(in_npz),
        stripe_masks=str(mask_npz),
        zsubmin=(None if args.zsubmin is None else float(args.zsubmin)),
        zsubmax=(None if args.zsubmax is None else float(args.zsubmax)),
    )

    print(f"[save] wrote {outfile}")


if __name__ == "__main__":
    main()