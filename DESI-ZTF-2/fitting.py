import os
import numpy as np
from astropy.constants import iau2015 as const
from astropy import units
from matplotlib import pyplot as plt
from matplotlib import gridspec
from astropy.cosmology import Planck18

def abs_mag_to_L(M):
    """
    Converts absolute magnitude to luminosity.
    
    Args:
        M : absolute magnitude
        L : luminosity in erg/s
    """
    L_0 = const.L_bol0.to(units.erg / units.s).value
    L = L_0 * 10.0 ** (-0.4 * M)
    return L


# https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3252S/abstract
def get_lfpars_shen20(z):
    """
    Returns the (gamma1, gamma2, L_star, phi_star) parameters
        of the Shen+20 paper for a given redshift.
    
    Args:
        z: redshift

    Returns:
        gamma1: gamma1 parameter
        gamma2: gamma2 parameter
        L_star: L_star parameter (in erg/s)
        phi_star: phi_star parameter

    Notes:
        Values from best-fit A, Table 4.
    """

    params_A = {
        "a0": 0.8569,
        "a1": -0.2614,
        "a2": 0.0200,
        "b0": 2.5375,
        "b1": -1.0425,
        "b2": 1.1201,
        "c0": 13.0088,
        "c1": -0.5759,
        "c2": 0.4554,
        "d0": -3.5426,
        "d1": -0.3936,
    }
    z_ref = 2  # see after Eq. 14
    
    # gamma1
    a0, a1, a2 = params_A["a0"], params_A["a1"], params_A["a2"]
    gamma1 = a0 * T0(1 + z) + a1 * T1(1 + z) + a2 * T2(1 + z)
    # gamma2
    b0, b1, b2 = params_A["b0"], params_A["b1"], params_A["b2"]
    gamma2 = 2 * b0 / (((1 + z) / (1 + z_ref)) ** b1 + ((1 + z) / (1 + z_ref)) ** b2)
    # L_star (Eq. 14 gives L_star in L_sun units)
    c0, c1, c2 = params_A["c0"], params_A["c1"], params_A["c2"]
    L_star = 10.0 ** (
        2 * c0 / (((1 + z) / (1 + z_ref)) ** c1 + ((1 + z) / (1 + z_ref)) ** c2)
    )
    L_sun = const.L_sun.to(units.erg / units.s).value 
    L_star *= L_sun
    # phi_star
    d0, d1 = params_A["d0"], params_A["d1"]
    phi_star = 10.0 ** (d0 * T0(1 + z) + d1 * T1(1 + z))

    return gamma1, gamma2, L_star, phi_star
    # Chebyshev polynomials
def T0(x):
    return 1

def T1(x):
    return x

def T2(x):
    return 2 * x ** 2 - 1


def get_lfpars(paper, z):

    assert paper in ["shen20"]

    if paper == "shen20":

        gamma1, gamma2, L_star, phi_star = get_lfpars_shen20(z)

    return gamma1, gamma2, L_star, phi_star


def L_to_M(L):
    """
    Converts absolute magnitude to luminosity.
    
    Args:
        M : absolute magnitude
        L : luminosity in erg/s
    """
    L_0 = const.L_bol0.to(units.erg / units.s).value
    return -2.5*numpy.log10(L/L0)

def get_phis(Ls, z, paper):
    gamma1, gamma2, L_star, phi_star = get_lfpars(paper, z)
    phis = phi_star / ((Ls / L_star) ** gamma1 + (Ls / L_star) ** gamma2)
    return phis

# class Phi_0(object):
    # def __init__(self, Nsamples=1000):
    #     self._x = numpy.random.uniform(0,1,Nsamples)

    # def call(self, alpha, beta, Lmin):
    #     x = self._x*Lmin**((a+b)/2+1)/((a+b)/2+1)
    #     L=(((a+b)/2+1)*x)**(1/(1+(a+b)/2))
    #     return -(Lmin**((a+b)/2+1)/((a+b)/2+1)) * (1/(L**((b-a)/2) + L**((-b+a)/2))).mean()

class phi(object):
    def __init__(self, Nsamples=1000):
        self._x = numpy.random.uniform(0,1,Nsamples)

    def __call__(self, _L, alpha, beta, Lmin):
        x = self._x[None,:]*Lmin**((alpha[:,None]+beta[:,None])/2+1)/((alpha[:,None]+beta[:,None])/2+1)
        L=(((alpha[:,None]+beta[:,None])/2+1)*x)**(1/(1+(alpha[:,None]+beta[:,None])/2))
        norm = -(Lmin**((alpha+beta)/2+1)/((alpha+beta)/2+1)) * (1/(L**((beta[:,None]-alpha[:,None])/2) + L**((-beta[:,None]-alpha[:,None])/2))).mean(axis=1)
        return 1/(_L**(-alpha) + _L**(-beta))/norm


class N_obs(object):
    def __init__(self, zmin, zmax, eff, Nsamples=1000):
        self.desi_fraction = 0.16
        self._x = numpy.random.uniform(0,1,Nsamples)
        self._y = numpy.random.normal(0,1,Nsamples)
        _, _, self.L_star, phi_star = get_lfpars_shen20((zmax+zmin)/2)
        self.phi_star_over_ln10 = phi_star/numpy.log(10)
        self.eff = eff
        self.Volume = self.desi_fraction*(Planck18.comoving_volume(zmax)-Planck18.comoving_volume(zmin))

    def __call__(self, x, alpha, beta, Lmin, k, mu, sigma):
        x = self._x*Lmin**((alpha+beta)/2+1)/((alpha+beta)/2+1)
        L=(((alpha+beta)/2+1)*x)**(1/(1+(alpha+beta)/2))
        M = L_to_M(L)
        m = self._y * sigma + M + x+ k + mu
        return -self.Volume*(Lmin**((alpha+beta)/2+1)/((alpha+beta)/2+1)) * self.phi_star_over_ln10* (self.eff(m)/(L**((beta-alpha)/2) + L**((-beta+alpha)/2))).mean()

class discovery_fraction(object):
    def __init__(self, eff, Nsamples=1000):
        self._y = numpy.random.normal(0,1,Nsamples)
        self.eff = eff

    def __call__(self, x, M, k, mu,,sigma):
        m = self._y[None,:] * sigma[:,None] + M[:,None] + x+ k[:,None] + mu[:,None]
        return self.eff(m).mean(axis=1)

class ln_posterior(object):
    def __init__(self, z, eff, Nsamples=1000):
        self._y = numpy.random.normal(0,1,Nsamples)
        self.eff = eff
        self.discovery_fraction = discovery_fraction(eff)
        self.phi = phi()
        self.N_obs = N_obs(z,eff)

    def __call__(self, x, mhat, k, mu,,sigma):
        nquasar=len(mhat)
        M = self._y[None, :] * sigma[:,None] + mhat[:,None] - x - k[:,None] - mu[:,None]
        df = self.discovery_fraction(x,M,k,mu,sigma)
        N_obs = self.N_obs(x,alpha,beta,Lmin,k,mu,sigma)
        L = abs_mag_to_L(M)
        phi = self,phi(L,alpha,beta,Lmin)
        integrand = self.eff(mhat)/df*phi
        maxintegrand = integrand.max()

        return jnp.log(maxinterand)  + jnp.log((integrand/maxintegrand).sum()) + nquasar*jnp.log(N_obs) - N_obs
