"""Functions needed for calculating synchrotron emission properties."""

import os
import h5py
import numpy as np
import json

import astropy.units as u
import astropy.constants as const
from scipy.special import gammainc, gamma
from scipy.integrate import quad
from math import factorial


# energy parameters
mp = const.m_p.cgs
me = const.m_e.cgs
c = const.c.cgs
mu = 1.4  # mean molecular weight
mpc2 = mp * c**2
mec2 = me * c**2
r0 = 2.8179403205e-15 * u.m  # classical electron radius
kB = const.k_B.cgs
hbar = const.hbar.cgs


def mom(e):
    """Normalised momentum as function of energy."""
    return np.sqrt((e + 1) ** 2 - 1)

def ekin(p):
    """Normalised kinetic energy as a function of normalised momentum"""
    return np.sqrt(p**2 + 1) - 1

def Ekin(p, mass):
    """Normalised kinetic energy as a function of normalised momentum"""
    return mass * c**2 * (np.sqrt(p**2 + 1) - 1)

def dp_dekin(p):
    """Derivative of normalised momentum wrt normalised kinetic energy."""
    return np.sqrt(p**2 + 1) / p

def dp_dEkin(p, mass):
    """Derivative of normalised momentum wrt kinetic energy."""
    return np.sqrt(p**2 + 1) / (p * mass * c**2)

def diff(p, diff_10GeV, mass=mp):
    """Diffusion coefficient as a function of normalised momentum."""
    return diff_10GeV * ((ekin(p) * mass * c**2 ).to(u.GeV) / (10 * u.GeV)).value**0.5

def t_accel(p, diff_10GeV, mass=mp, vwind=1e8 * u.cm / u.s):
    """Acceleration time as a function of normalised momentum."""
    return 8 * diff(p, diff_10GeV, mass) / vwind**2

def standoff_radius(vstar):
    """Standoff radius as a function of stellar velocity. Pre-computed already."""
    R0 = 1.02710 * u.pc if vstar == 40.0 * (u.km / u.s) else 0.42922 * u.pc
    return R0

def t_diff(p, vstar, diff_10GeV, mass=mp):
    """Diffusion time as a function of normalised momentum."""
    R0 = standoff_radius(vstar)
    return R0**2 / diff(p, diff_10GeV, mass)

def t_adv(vel, vstar):
    """Advection time as a function of stellar velocity."""
    R0 = standoff_radius(vstar)
    return R0 / vel

# losses for protons
def b_hadr(p, rho):
    """Hadronic loss coefficient, depends on normalised momentum.

    Dimensionless as it is normalised by time scaling factor tau0."""
    # hadronic loss parameters
    sigma_pp = 44 * u.mbarn  # total pion cross section
    Kp = 0.5  # inelasticity
    B_hadr = (c * sigma_pp * Kp * rho / (mu * mp)).cgs  # hadronic loss coefficient
    hadr_loss =  B_hadr * Ekin(p, mass=mp) * dp_dEkin(p, mass=mp)
    hadr_loss = np.where(np.array(p) < 0.8, 0, hadr_loss)
    return hadr_loss

def b_coul_p_approx(p, rho):
    """Coulomb loss coefficient for protons, depends on normalised momentum.

    This is given as Eq. 12 in Girichidis+2019.

    Dimensionless as it is normalised by time scaling factor tau0."""
    coul_coeff = -1e-18 * u.erg * u.cm**3 * u.s**-1
    n_e = 0.88 * rho / (mu * mp)
    B_coul = (coul_coeff * n_e / c / (mp * c)).cgs
    return B_coul * (1 + p**-1.9)

def b_coul_p(p, rho):
    """
    Coulomb loss for protons, obtained from Gould(1972a).
    """
    sigma_thom = 0.665 * u.barn  # thomson cross section (from NIST)
    n_e = 0.88 * rho / (mu * mp)
    plasma_freq = np.sqrt(4 * np.pi * n_e * r0 * c**2).cgs
    lor_gamma = np.sqrt(1 + p**2)
    lor_beta = p / lor_gamma

    prefac = (3 * sigma_thom * n_e * mec2 * c / (2 * lor_beta)).cgs
    log_term = np.log(
        2 * lor_gamma * lor_beta**2 * mec2 / (hbar * plasma_freq)
    )

    return (prefac * (log_term - 0.5 * lor_beta**2) * dp_dEkin(p, mass=mp)).to(1 / u.s)

def b_adiab(p, vel, vstar):
    """Adiabatic loss coefficient, depends on normalised momentum.

    This is given as Eq. 8 in Girichidis+2019.

    Dimensionless as it is normalised by time scaling factor tau0."""
    t_adv_val = t_adv(vel, vstar)
    B_adiab = (1 / t_adv_val).cgs
    return B_adiab * p

# losses for electrons

def b_synch(p, B):
    """Synchrotron loss coefficient for electrons, depends on normalised momentum.

    Dimensionless as it is normalised by time scaling factor tau0."""
    sigma_thom = 0.665 * u.barn  # thomson cross section (from NIST)
    u_mag = (B**2 / (2 * const.mu0)).to(
        u.erg / u.cm**3
    )  # magnetic field energy density
    B_synch = (4 / 3 * sigma_thom * c * u_mag).cgs
    return (B_synch * dp_dEkin(p, mass=me)).to(1 / u.s)

def b_ic_photon(p):
    """Inverse Compton loss coefficient from background photons, depends on normalised momentum.

    Dimensionless as it is normalised by time scaling factor tau0."""
    sigma_thom = 0.665 * u.barn  # thomson cross section (from NIST)
    u_phot = (0.5 * u.eV / u.cm**3).to(
        u.erg / u.cm**3
    )  # photon energy density from ISRF, Mathis 1983
    B_ic_phot = (4 / 3 * sigma_thom * c * u_phot).cgs
    return (B_ic_phot * dp_dEkin(p, mass=me)).to(1 / u.s)

def b_ic_star(p, Tstar, r_pos):
    """Inverse Compton loss coefficient from stellar photons, depends on normalised momentum.

    Obtained from Eq. 36 from Khangulyan et al. (2014), assuming isotropic scatterin of stellar photons.

    Dimensionless as it is normalised by time scaling factor tau0."""
    Rstar = 15 * u.Rsun  # stellar radius, assumed to be 20 Rsun for O star
    dil_factor = ((Rstar / (2 * r_pos))**2).to_value(u.dimensionless_unscaled)  # dilution factor for stellar radiation field
    coeff = dil_factor * (2 * r0**2 * kB**2 * me / (np.pi * hbar**3) * Tstar**2).cgs # should be in units of 1/s, but we will normalise it by tau0 later, so we can ignore the units for now
    
    u_star = (4 * Ekin(p, mass=me) / mec2 * kB * Tstar / mec2).to_value(u.dimensionless_unscaled)  # dimensionless parameter for IC scattering
    c_iso = 4.62
    F_iso = c_iso * u_star * np.log(1 + 0.722 * u_star / c_iso) / (1 + (c_iso * u_star) / 0.822)

    return (coeff * F_iso * dp_dekin(p)).to(1 / u.s)

def b_coul_e(p, rho):
    """
    Coulomb scattering for electrons. Obtained from Gould (1972b).
    """
    n_e = 0.88 * rho / (mu * mp)
    plasma_freq = np.sqrt(4 * np.pi * n_e * r0 * c**2).cgs
    sigma_thom = 0.665 * u.barn  # thomson cross section (from NIST)
    lor_fac = np.sqrt(1 + p**2)
    lor_beta = p / lor_fac

    prefac = (3 * sigma_thom * n_e * mec2 * c / (2 * lor_beta) ).cgs
    log_term = np.log(
        mec2 * lor_beta * np.sqrt(lor_fac-1) / (hbar * plasma_freq)
    )
    second_log_term = np.log(2) * (
        lor_beta**2 / 2 + 1 / lor_fac
    ) + 0.5 + ((lor_fac - 1) / (4 * lor_fac))**2

    return prefac * (log_term - second_log_term) * dp_dEkin(p, mass=me)

def tcool_proton(p, rho):
    t_hadr = 1 / np.abs(b_hadr(p, rho))
    t_coul = 1 / np.abs(b_coul_p(p, rho))

    return (1 / t_hadr + 1 / t_coul)**-1 

def tcool_electron(p, rho, B, Tstar, vstar):
    t_synch = 1 / np.abs(b_synch(p, B))
    t_ic_phot = 1 / np.abs(b_ic_photon(p))
    r_pos = standoff_radius(vstar)
    t_ic_star = 1 / np.abs(b_ic_star(p, Tstar, r_pos))
    t_coul = 1 / np.abs(b_coul_e(p, rho))

    return (1 / t_synch + 1 / t_ic_phot + 1 / t_ic_star + 1 / t_coul)**-1

def get_integral(alpha_inj, p_arr, pmax):
    s = 3 - alpha_inj
    t = p_arr / pmax
    # For negative s, compute upper incomplete gamma through integration
    def upper_gamma_neg(s, t, t_series_switch=1e-2, nterms=30):
        """Compute upper incomplete gamma Gamma(s, t) for negative, non-integer s.

        scipy.integrate.quad is used directly for t >~ t_series_switch, where it is
        accurate. For smaller t (as happens when p_arr/pmax << 1, e.g. very high
        pmax cases), quad's adaptive integration of the near-singular, slowly
        decaying integrand u^(s-1)*exp(-u) becomes unreliable or fails to converge
        (it can even return unphysical negative values there). In that regime we
        instead use Gamma(s, t) = Gamma(s) - gamma_lower(s, t), with gamma_lower(s, t)
        evaluated via its (rapidly convergent for small t) power series
            gamma_lower(s, t) = t^s * sum_{k=0}^inf (-t)^k / (k! * (s + k))
        This matches quad to high precision everywhere quad itself converges, and
        stays well-behaved below that.
        """
        is_scalar = np.isscalar(t)
        t_arr = np.atleast_1d(t).astype(float)

        gamma_s = gamma(s)
        results = np.empty_like(t_arr)

        small = t_arr < t_series_switch
        if np.any(small):
            k = np.arange(nterms)
            # sum_k (-t)^k / (k! * (s+k)), broadcast over t and k
            terms = (-t_arr[small, None]) ** k / np.array([factorial(ki) for ki in k]) / (s + k)
            gamma_lower = t_arr[small] ** s * terms.sum(axis=1)
            results[small] = gamma_s - gamma_lower

        large = ~small
        for idx in np.where(large)[0]:
            result, _ = quad(lambda u: u**(s-1) * np.exp(-u), t_arr[idx], np.inf, epsabs=1e-10)
            results[idx] = result

        if is_scalar:
            return results[0]
        else:
            return results

    gamma_value = upper_gamma_neg(s, t)
    return gamma_value

# steady state distribution of protons, assuming hadronic, coulomb, and advection losses.
# assume a power-law + exponential cutoff injection spectrum
# steady state spectrum with units of 1 / s
def f_p_0(alpha_inj, p_arr, rho, vel, pmax, vstar):
    """Return the steady state spectrum for protons with given injection index."""
    # NB: need to multiply by gamma() since gammainc() is the regularised incomplete gamma function, which divides by gamma() to normalise it.
    cutoff_part = get_integral(alpha_inj, p_arr, pmax)
    # if not np.isscalar(p_arr):
    #     cutoff_part = cutoff_part[:,None,None,None,None]
    # print(cutoff_part)
    mom_part = (p_arr / pmax)**(-2) * pmax**(1 - alpha_inj) * cutoff_part
    loss_part = np.abs(b_hadr(p_arr, rho) + b_coul_p(p_arr, rho) + b_adiab(p_arr, vel, vstar))
    mom_rescale = ((mp * c)**(-alpha_inj + 1))
    return mom_part / loss_part * mom_rescale

def f_e_0(alpha_inj, p_arr, rho, B, Tstar, vel, r_pos, pmax, vstar):
    """Return the steady state spectrum for electrons with given injection index."""
    # NB: need to multiply by gamma() since gammainc() is the regularised incomplete gamma function, which divides by gamma() to normalise it.
    cutoff_part = get_integral(alpha_inj, p_arr, pmax)
    # if not np.isscalar(p_arr):
    #     cutoff_part = cutoff_part[:,None,None,None,None]
    # print(cutoff_part)
    mom_part = (p_arr / pmax)**(-2) * pmax**(1 - alpha_inj) * cutoff_part
    loss_part = np.abs(b_synch(p_arr, B) + b_ic_photon(p_arr) + b_ic_star(p_arr, Tstar, r_pos) + b_coul_e(p_arr, rho) + b_adiab(p_arr, vel, vstar))
    mom_rescale = (me * c)**(-alpha_inj + 1)
    return mom_part / loss_part * mom_rescale

def ep_ratio():
    """Calculate the electron to proton ratio at 10 GeV, based on Cummings et al. 2016."""
    # parameters for taking care of normalisation of p-> e spectrum
    # the observed spectrum is evaluated at 10 GeV, so we normalise energy
    # then transform to normalised momentum
    Kep = 0.01  # Cummings et al 2016
    pp0 = ((10 * u.GeV) / c / (mp * c)).to_value(u.dimensionless_unscaled)
    pe0 = ((10 * u.GeV) / c/ (me * c)).to_value(u.dimensionless_unscaled)
    # pp0 = np.sqrt((ep0 + 1) ** 2 - 1)
    # pe0 = np.sqrt((ee0 + 1) ** 2 - 1)
    print(f"ep ratio parameters: {Kep:.2f}, {pp0:.3f}, {pe0:.3f}")
    return Kep * (me / mp), pp0, pe0


# below are functions to calculation synchrotron emitting electrons
def synch_kernel(x):
    """Synchrotron kernal, as in Aharonian et al. 2010."""
    exp_factor = (1 + 0.884 * x ** (2 / 3) + 0.471 * x ** (4 / 3)) / (
        1 + 1.64 * x ** (2 / 3) + 0.974 * x ** (4 / 3)
    )
    return 2.15 * x ** (1 / 3) * (1 + 3.06 * x) ** (1 / 6) * exp_factor * np.exp(-x)

def crit_freq(Blos, pele):
    """Critical frequency as a function of magnetic field / microG and electron energy / GeV"""
    # NB: here we take normalised momentum (normalised over mass)
    return 0.5 * (Blos / (1 * u.uG)) * (np.sqrt(1 + pele**2) / 1e4) ** 2 * u.GHz