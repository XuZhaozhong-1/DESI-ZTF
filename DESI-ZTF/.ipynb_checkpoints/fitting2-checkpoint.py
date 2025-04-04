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

def MgtoLg(Mg_star):
    # Absolute magnitude of the Sun in the g-band
    M_g_sun = 5.12
    
    # Solar luminosity in the g-band (1 L_sun for simplicity)
    L_g_sun = 1.0  # In solar units
    
    # Convert absolute magnitude to luminosity in g-band (in solar units)
    Lg_star = L_g_sun * 10**(-0.4 * (Mg_star - M_g_sun))
    return Lg_star
    
def get_lfpars_pd17(z):
    params_PLE_LEDE2 = {
        "alpha":-3.16,
        "beta":-1.49,
        "c1":-0.59,
        "c2":1.37,
        "zp":2.05,
        "phi_star_const":10**-5.7,
        "Mg_star_const":-25.95,
    }
    zp =  params_PLE_LEDE2["zp"]
    c1 =  params_PLE_LEDE2["c1"]
    c2 =  params_PLE_LEDE2["c2"]
    zp =  params_PLE_LEDE2["zp"]
    alpha =  params_PLE_LEDE2["alpha"]
    beta =  params_PLE_LEDE2["beta"]
    phi_star_const =  params_PLE_LEDE2["phi_star_const"]
    Mg_star_const =  params_PLE_LEDE2["Mg_star_const"]
    phi_star = jnp.where(z < zp, phi_star_const,phi_star_const*10**(c1*(z-zp)))
    Mg_star = Mg_star_const - c2*2.5*jnp.log10(1+z-jnp.sqrt(1+z))
    Lg_star = MgtoLg(Mg_star)
    gamma1 = -(alpha+1)
    gamma2 = -(beta+1)
    phi_star_prime = 2.5*phi_star
    #return alpha,beta,phi_star,Mg_star
    return gamma1,gamma2,Lg_star,phi_star_prime
    
def get_phi_pd17(Mg,z):
    alpha,beta,phi_star,Mg_star = get_lfpars_pd17(z)
    phis = phi_star/(10**(0.4*(alpha+1)*(Mg-Mg_star))+10**(0.4*(beta+1)*(Mg-Mg_star)))
    return phis

def T0(x):
    return 1

def T1(x):
    return x

def T2(x):
    return 2 * x ** 2 - 1


def get_lfpars(paper, z):

    #assert paper in ["shen20"]

    if paper == "shen20":

        gamma1, gamma2, L_star, phi_star = get_lfpars_shen20(z)
        
    if paper == "pd17":
        
        gamma1, gamma2, L_star, phi_star = get_lfpars_pd17(z)

    return gamma1, gamma2, L_star, phi_star


def L_to_M(L):
    """
    Converts absolute magnitude to luminosity.
    
    Args:
        M : absolute magnitude
        L : luminosity in erg/s
    """
    L_0 = const.L_bol0.to(units.erg / units.s).value
    return -2.5*numpy.log10(L/L_0)

def get_phis(Ls, z, paper):
    gamma1, gamma2, L_star, phi_star = get_lfpars(paper, z)
    #gamma1 -= 0.4
    #gamma2 += 0.1
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
    def __init__(self, Nsamples=1000,key=jax.random.PRNGKey(0)):
        self._x = jax.random.uniform(key,(Nsamples,))

    def __call__(self, _L, alpha, beta, Lmin):
        x = self._x*Lmin**((alpha+beta)/2+1)/((alpha+beta)/2+1)
        L=(((alpha+beta)/2+1)*x)**(1/(1+(alpha+beta)/2))
        norm = -(Lmin**((alpha+beta)/2+1)/((alpha+beta)/2+1)) * (1/(L**((beta-alpha)/2) + L**((-beta-alpha)/2))).mean()
        return 1/(_L**(-alpha) + _L**(-beta))/norm
	
def phi_new(L,alpha,beta,Lmin):
    norm = integral_Lmin_Lmax(Lmin, jnp.inf, beta, alpha)
    phi = 1/(L**(-alpha)+L**(-beta))/norm
    return phi
        
def hyp2f1(x, z, buf=10):
    # choose buf >> abs(x)
    ns = jnp.arange(1,buf,dtype='int')
    terms = x/(x+ns)*z**ns
    ans = 1 + terms.sum()
    return ans
    
def integral(L, alpha, beta):
    def case_L_1(L, alpha, beta):
        return (
            0.5 * (1 + alpha) / (alpha - beta) *
            (jax.scipy.special.digamma(0.5 * (1 + alpha) / (alpha - beta) + 0.5) -
             jax.scipy.special.digamma(0.5 * (1 + alpha) / (alpha - beta))) /
            (1 + alpha)
        )

    def case_L_small(L, alpha, beta):
        return L**(alpha + 1) * hyp2f1((1 + alpha) / (alpha - beta), -L**(alpha - beta)) / (1 + alpha)

    def case_error(L, alpha, beta):
        return jnp.nan  # Handle cases where conditions are not met

    condition_L1 = (L == 1)
    condition_L_small = (jnp.abs(L**(alpha - beta)) < 1)

    # Use lambdas to delay execution and pass arguments properly
    ans = lax.cond(
        condition_L1,
        lambda _: case_L_1(L, alpha, beta),  # Execute case_L_1 if L == 1
        lambda _: lax.cond(
            condition_L_small,
            lambda _: case_L_small(L, alpha, beta),  # Execute case_L_small if condition_L_small is True
            lambda _: case_error(L, alpha, beta),  # Execute case_error otherwise
            None  # No operand needed
        ),
        None  # No operand needed for the outer cond
    )

    return ans

def integral_Lmin_Lmax(Lmin, Lmax, alpha, beta):
    def cond_below_one(Lmin, alpha, beta):
        return -integral(Lmin, beta, alpha)

    def cond_above_one(Lmin, alpha, beta):
        return -integral(Lmin, alpha, beta)

    def integral_Lmax_below_one(Lmax, alpha, beta):
        return integral(Lmax, beta, alpha)

    def integral_Lmax_above_one(Lmax, alpha, beta):
        return integral(Lmax, alpha, beta)

    Lmin_cond = Lmin < 1
    Lmax_cond = Lmax <= 1
    Lmax_inf_cond = jnp.isinf(Lmax)

    ans = lax.cond(
        Lmin_cond,
        lambda _: cond_below_one(Lmin, alpha, beta),
        lambda _: cond_above_one(Lmin, alpha, beta),
        operand=None
    )

    ans += lax.cond(
        Lmax_inf_cond,
        lambda _: 0.0,
        lambda _: lax.cond(
            Lmax_cond,
            lambda _: integral_Lmax_below_one(Lmax, alpha, beta),
            lambda _: integral_Lmax_above_one(Lmax, alpha, beta),
            operand=None
        ),
        operand=None
    )

    ans += lax.cond(
        Lmin_cond & (Lmax > 1),
        lambda _: -integral(1, alpha, beta) + integral(1, beta, alpha),
        lambda _: 0.0,
        operand=None
    )

    #if check:
    #    print(ans, jax.scipy.integrate.quad(lambda L: 1/(L**-alpha+L**-beta), Lmin, Lmax))

    return ans

def integral_numerical(Lmin,Lmax,alpha,beta,Nsamples=100):
    new_L = jnp.linspace(Lmin,Lmax,Nsamples)
    integrand = 1/(new_L**-alpha+new_L**-beta)
    integral = jnp.trapezoid(integrand,new_L)
    return integral

def test():
    n=1
    beta = -(0.411+1)
    alpha = -(2.487+1)

    # beta = (1+2*alpha*(1+n))/(n+1)
    print(alpha,beta)
    Lmin=.1
    # Lmax=10
    # print(integral_Lmin_Lmax(Lmin, Lmax, alpha, beta, check=True), integral_Lmin_Lmax(Lmin, Lmax, alpha, beta, approx=False))
    print(integral_Lmin_Lmax(Lmin, jnp.inf, alpha, beta, check=True), integral_Lmin_Lmax(Lmin, jnp.inf, alpha, beta, approx=False))

#test()

class denominator(object):
    def __init__(self,zmin,zmax):
        self.zmean = (zmin+zmax)/2
        self.gamma1, self.gamma2, self.L_star, self.phi_star = get_lfpars_shen20(self.zmean)
        self.alpha = -(self.gamma1+1)
        self.beta = -(self.gamma2+1)
        self.phi_star_over_ln10 = self.phi_star/jnp.log(10)

    def __call__(self,Lmin):
        ans = self.phi_star_over_ln10 * integral_Lmin_Lmax(Lmin,jnp.inf,self.beta,self.alpha)
        return ans

class Prob_m(object):
    def __init__(self,zmin,zmax):
        self.zmean = (zmin+zmax)/2
        self.gamma1, self.gamma2, self.L_star, self.phi_star = get_lfpars_shen20(self.zmean)
        self.alpha = -(self.gamma1+1)
        self.beta = -(self.gamma2+1)

    def __call__(self,mhat,x,k,mu,denominator):
        L = abs_mag_to_L(mhat-k-x-mu)/self.L_star
        numerator = 0.4*self.phi_star/(L**(-self.alpha-1)+L**(-self.beta-1))
        return numerator / denominator

class Prob_det(object):
    def __init__(self,zmin,zmax):
        self.zmean = (zmin+zmax)/2
        self.gamma1, self.gamma2, self.L_star, self.phi_star = get_lfpars_shen20(self.zmean)
        self.alpha = -(self.gamma1+1)
        self.beta = -(self.gamma2+1)

    def __call__(self,m0,b,x,k,mu,Lmin,denominator):
        L1 = abs_mag_to_L(m0-k.mean()-x-mu.mean())/self.L_star
        const1 = (1/denominator)*self.phi_star/jnp.log(10)*10**(b*(m0-x-k.mean()-mu.mean())/2.5)
        const2 = (1/denominator)*self.phi_star/jnp.log(10)
        term1 = const1*integral_Lmin_Lmax(Lmin, L1, self.beta+b, self.alpha+b)
        term2 = const2*integral_Lmin_Lmax(L1,jnp.inf,self.beta, self.alpha)
        ans = term1 + term2
        #print(const1)
        #print(const2)
        return ans


class ln_posterior(object):
    def __init__(self,eff,fraction=0.014,zmin=2.3,zmax=2.4,Lmin=0.01):
        self.Lmin = Lmin
        self.eff = eff
        self.gamma1, self.gamma2, self.L_star, phi_star = get_lfpars_shen20((zmin+zmax)/2)
        self.alpha = -(self.gamma1+1)
        self.beta = -(self.gamma2+1)
        self.denominator = denominator(zmin,zmax)
        self.Prob_m = Prob_m(zmin,zmax)
        self.Prob_det = Prob_det(zmin,zmax)
        self.Volume = (Planck18.comoving_volume(zmax)-Planck18.comoving_volume(zmin)).value

    def __call__(self,m0,b,x,mhat,k,mu,fraction):
        nsamples = len(mhat)
        #Lmin = 1e-4
        Lmin = abs_mag_to_L(mhat.max()-k.mean()-x-mu.mean())/self.L_star
        denominator_term = self.denominator(Lmin)
        denominator_term_Ntot = self.denominator(1e-4)
        Prob_detection = self.Prob_det(m0,b,x,k,mu,Lmin,denominator_term)
        Prob_mag = self.Prob_m(mhat,x,k,mu,denominator_term)
        efficiency = self.eff(mhat,b,m0)
        N_Total = self.Volume*denominator_term_Ntot*fraction
        #print(Prob_detection)
        #print(self.Volume)
        #print(denominator_term)
        #print(fraction)
        #print(N_Total)
        term1 = -Prob_detection * N_Total
        term2 = jnp.sum(jnp.log(Prob_mag))
        term3 = jnp.sum(jnp.log(efficiency))
        #term4 = nsamples*jnp.log(N_Total)

        return term1 + term2 + term3
        