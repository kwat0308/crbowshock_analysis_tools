"""Functions needed for calculating synchrotron emission properties."""

import os
import h5py
import numpy as np
import json

import astropy.units as u
import astropy.constants as const
from scipy.special import gammainc, gamma


# energy parameters
mp = const.m_p.cgs
me = const.m_e.cgs
c = const.c.cgs
mu = 1.4  # mean molecular weight
mpc2 = mp * c**2
mec2 = me * c**2
r0 = const.physical_constants["classical electron radius"][0] * u.m
kB = const.k_B.cgs
hbar = const.hbar.cgs


def mom(e):
    """Normalised momentum as function of energy."""
    return np.sqrt((e + 1) ** 2 - 1)

def ekin(p):
    """Normalised kinetic energy as a function of normalised momentum"""
    return np.sqrt(p**2 + 1) - 1

def dp_dekin(p):
    """Derivative of normalised momentum wrt normalised kinetic energy."""
    return 2 * (ekin(p) + 1) / p

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

def t_adv(vstar):
    """Advection time as a function of stellar velocity."""
    R0 = standoff_radius(vstar)
    return R0 / vstar

# losses for protons
def b_hadr(p, rho):
    """Hadronic loss coefficient, depends on normalised momentum.

    Dimensionless as it is normalised by time scaling factor tau0."""
    # hadronic loss parameters
    sigma_pp = 44 * u.mbarn  # total pion cross section
    Kp = 0.5  # inelasticity
    B_hadr = (c * sigma_pp * Kp * rho / (mu * mp)).cgs  # hadronic loss coefficient
    return B_hadr * ekin(p) * dp_dekin(p)

def b_coul(p, rho):
    """Coulomb loss coefficient for protons, depends on normalised momentum.

    This is given as Eq. 12 in Girichidis+2019.

    Dimensionless as it is normalised by time scaling factor tau0."""
    coul_coeff = -1e-18 * u.erg * u.cm**3 * u.s**-1
    n_e = 0.88 * rho / (mu * mp)
    B_coul = (coul_coeff * n_e / c / (mp * c)).cgs
    return B_coul * (1 + p**-1.9)

def b_adiab(p, vstar):
    """Adiabatic loss coefficient, depends on normalised momentum.

    This is given as Eq. 8 in Girichidis+2019.

    Dimensionless as it is normalised by time scaling factor tau0."""
    t_adv_val = t_adv(vstar)
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
    B_synch = (4 / 3 * sigma_thom * c * u_mag / (me * c**2)).cgs
    return B_synch * p**2 * dp_dekin(p)

def b_ic_photon(p):
    """Inverse Compton loss coefficient from background photons, depends on normalised momentum.

    Dimensionless as it is normalised by time scaling factor tau0."""
    sigma_thom = 0.665 * u.barn  # thomson cross section (from NIST)
    u_phot = (0.5 * u.eV / u.cm**3).to(
        u.erg / u.cm**3
    )  # photon energy density from ISRF, Mathis 1983
    B_ic_phot = (4 / 3 * sigma_thom * c * u_phot / (me * c**2)).cgs
    return B_ic_phot * p**2 * dp_dekin(p)

def b_ic_star(p, Tstar):
    """Inverse Compton loss coefficient from stellar photons, depends on normalised momentum.

    Obtained from Eq. 36 from Khangulyan et al. (2014), assuming isotropic scatterin of stellar photons.

    Dimensionless as it is normalised by time scaling factor tau0."""
    coeff = (2 * r0**2 * kB**2 * me / (4 * np.pi**2 * hbar**3) * Tstar**2).cgs # should be in units of 1/s, but we will normalise it by tau0 later, so we can ignore the units for now
    u_star = (4 * ekin(p) * kB * Tstar / (mec2)).to_value(u.dimensionless_unscaled)  # dimensionless parameter for IC scattering
    c_iso = 4.62
    F_iso = c_iso * u_star * np.log(1 + 0.722 * u_star / c_iso) / (1 + (c_iso * u_star) / 0.822)

    return coeff * F_iso * p**2 * dp_dekin(p)

def tcool_proton(p, rho):
    t_hadr = 1 / np.abs(b_hadr(p, rho))
    t_coul = 1 / np.abs(b_coul(p, rho))

    return (1 / t_hadr + 1 / t_coul)**-1 

def tcool_electron(p, B, Tstar):
    t_synch = 1 / np.abs(b_synch(p, B))
    t_ic_phot = 1 / np.abs(b_ic_photon(p))
    t_ic_star = 1 / np.abs(b_ic_star(p, Tstar))

    return (1 / t_synch + 1 / t_ic_phot + 1 / t_ic_star)**-1

# steady state distribution of protons, assuming hadronic, coulomb, and advection losses.
# assume a power-law + exponential cutoff injection spectrum
# steady state spectrum with units of 1 / s
def f_p_0(alpha_inj, p_arr, rho, vel, pmax):
    """Return the steady state spectrum for protons with given injection index."""
    # NB: need to multiply by gamma() since gammainc() is the regularised incomplete gamma function, which divides by gamma() to normalise it.
    cutoff_part = gammainc(3 - alpha_inj, p_arr / pmax) * gamma(3 - alpha_inj)
    mom_part = (p_arr / pmax)**(-2) * pmax**(1 - alpha_inj) * cutoff_part
    loss_part = b_hadr(p_arr, rho) + b_coul(p_arr, rho) + b_adiab(p_arr, vel)
    return -1 * mom_part / loss_part

def f_e_0(alpha_inj, p_arr, B, Tstar, vel, pmax):
    """Return the steady state spectrum for electrons with given injection index."""
    # NB: need to multiply by gamma() since gammainc() is the regularised incomplete gamma function, which divides by gamma() to normalise it.
    cutoff_part = gammainc(3 - alpha_inj, p_arr / pmax) * gamma(3 - alpha_inj)
    mom_part = (p_arr / pmax)**(-2) * pmax**(1 - alpha_inj) * cutoff_part
    loss_part = b_synch(p_arr, B) + b_ic_photon(p_arr) + b_ic_star(p_arr, Tstar) + b_adiab(p_arr, vel)
    return -1 * mom_part / loss_part

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
    return Kep * (mp / me), pp0, pe0


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