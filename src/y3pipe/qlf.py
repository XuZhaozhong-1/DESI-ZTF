# src/y3pipe/qlf.py
import os
import numpy as np
from astropy.constants import iau2015 as const
from astropy import units
from matplotlib import pyplot as plt
from matplotlib import gridspec
from astropy.cosmology import Planck18
import jax
import scipy.integrate as integrate
import scipy.special
import scipy.stats
import numpy
import jax.lax as lax
import jax.numpy as jnp
from astropy.constants import L_sun
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import softplus

class ln_likelihood_bin(object):
    def __init__(self, zmin, zmax):
        self.Volume = (Planck18.comoving_volume(zmax) - Planck18.comoving_volume(zmin)).value

    def __call__(self, m0, b, m, sigma_m, k, sigma_k, mu,
                 log_Nphi, log_Lstar, alpha, beta):
        # weighted k summary
        w_k    = 1.0 / (sigma_k**2)
        k_mean = jnp.sum(k * w_k) / jnp.sum(w_k)
        # sigma_k_mean kept if you want it for diagnostics:
        # sigma_k_mean = 1.0 / jnp.sqrt(jnp.sum(w_k))

        # absolute magnitude with your mean-k and sigma_m approximation
        M0 = m - mu - (jnp.log(10)/2.5) * (sigma_m.mean()**2) * alpha - k_mean

        # selection
        log_efficiency = -softplus(b * (m - m0))

        # convolution correction (your form)
        exponent = - ((jnp.log(10)/2.5) * alpha * sigma_m.mean()**2)**2 / (2.0 * (sigma_m.mean()**2))

        # log L0 with your convention (log_Lstar already natural log)
        lnL0 = jnp.log(const.L_bol0.to(units.erg/units.s).value) - 0.4 * M0 * jnp.log(10.0) - log_Lstar

        # denominator of double power-law (kept with your signs)
        log_den = logsumexp(
            jnp.stack([-(alpha + 1.0) * lnL0, -(beta + 1.0) * lnL0], axis=0),
            axis=0
        )

        # intensity in log space
        log_I = log_Nphi + log_efficiency + exponent - log_den + jnp.log(jnp.log(10.0)/2.5)

        # ensure m is sorted for trapezoid
        order = jnp.argsort(m)
        m_ord = m[order]
        
        log_I_ord = log_I[order]
        I_ord = jnp.exp(log_I_ord)
        term1 = -jnp.trapezoid(I_ord, m_ord)

        # data (point-process) term — your chosen form
        term2 = jnp.sum(log_I)

        return term1 + term2