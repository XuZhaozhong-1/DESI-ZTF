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
    def __init__(self, sigma_m, k, sigma_k, zmin, zmax, order=0, sigma_coeff=None, mu=None, m_ref=22.0):
        w_k = 1.0 / (sigma_k**2)
        self.k_mean = jnp.sum(k * w_k) / jnp.sum(w_k)
        if mu is None:
            self.mu = jnp.asarray(Planck18.distmod(0.5*(zmin+zmax)).value)
        else:
            self.mu = mu
        self.order = int(order)
        self.m_ref = float(m_ref)

        if self.order == 0:
            self.sigma_const = jnp.mean(sigma_m)   # scalar
            self.sigma_coeff = None
        else:
            if sigma_coeff is None:
                raise ValueError("order>=1 requires sigma_coeff (polyfit coefficients).")
            self.sigma_coeff = jnp.asarray(sigma_coeff)

    def sigma_m_of_m(self, m):
        if self.order == 0:
            return jnp.ones_like(m) * self.sigma_const
        x = m - self.m_ref
        log10_sigma = jnp.polyval(self.sigma_coeff, x)
        return jnp.power(10.0, log10_sigma)

    def __call__(self, m0, b, m, log_Nphi, log_Lstar, alpha, beta):
        c = jnp.log(10.0) / 2.5

        sigma_m = self.sigma_m_of_m(m)

        M0 = m - self.mu - self.k_mean - c * (sigma_m**2) * alpha
        lnL0 = -0.4 * M0 * jnp.log(10.0) - log_Lstar

        log_den = logsumexp(
            jnp.stack([-(alpha + 1.0) * lnL0, -(beta + 1.0) * lnL0], axis=0),
            axis=0
        )

        exponent = -0.5 * (c**2) * (sigma_m**2) * (alpha**2)
        log_eff = -softplus(b * (m - m0))

        log_I = log_Nphi + log_eff + exponent - log_den + jnp.log(c)

        order = jnp.argsort(m)
        m_ord = m[order]
        I_ord = jnp.exp(log_I[order])

        term1 = -jnp.trapezoid(I_ord, m_ord)
        term2 = jnp.sum(log_I)
        return term1 + term2