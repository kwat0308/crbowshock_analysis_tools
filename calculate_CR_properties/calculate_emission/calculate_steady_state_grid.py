import numpy as np
import os
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const

from scipy.integrate import quad
from tqdm import tqdm
from tqdm.contrib import itertools

from joblib import delayed, Parallel

import h5py

data_dir = "../../data"

# energy parameters
mp = const.m_p.cgs
me = const.m_e.cgs
c = const.c.cgs
mu = 1.4  # mean molecular weight


def mom(e):
    """Normalised momentum as function of energy."""
    return np.sqrt((e + 1) ** 2 - 1)


pp_arr = np.logspace(0.8, 4, 200) * u.GeV / c  # energy grid for protons
pp_arr = (pp_arr / (mp * c)).to_value(
    u.dimensionless_unscaled
)  # normalised momentum wrt protons
print(
    "Minimum & maximum of normalised momentum (protons): ", pp_arr.min(), pp_arr.max()
)

pe_arr = np.logspace(-0.25, 4, 200) * u.GeV / c # energy grid for electrons
pe_arr = (pe_arr / (me * c)).to_value(
    u.dimensionless_unscaled
)  # normalised momentum wrt protons
print(
    "Minimum & maximum of normalised momentum (electrons): ", pe_arr.min(), pe_arr.max()
)

# parameters for taking care of normalisation of p-> e spectrum
# the observed spectrum is evaluated at 10 GeV, so we normalise energy
# then transform to normalised momentum
Kep = 0.01  # Cummings et al 2016
pp0 = ((10 * u.GeV) / c / (mp * c)).to_value(u.dimensionless_unscaled)
pe0 = ((10 * u.GeV) / c/ (me * c)).to_value(u.dimensionless_unscaled)
# pp0 = np.sqrt((ep0 + 1) ** 2 - 1)
# pe0 = np.sqrt((ee0 + 1) ** 2 - 1)
print(f"ep ratio parameters: {Kep:.2f}, {pp0:.3f}, {pe0:.3f}")

# raise Exception()


def ekin(p):
    """Normalised kinetic energy as a function of normalised momentum"""
    return np.sqrt(p**2 + 1) - 1


def dp_dekin(p):
    """Derivative of normalised momentum wrt normalised kinetic energy."""
    return 2 * (ekin(p) + 1) / p


def b_hadr(p, rho):
    """Hadronic loss coefficient, depends on normalised momentum.

    Dimensionless as it is normalised by time scaling factor tau0."""
    # hadronic loss parameters
    sigma_pp = 44 * u.mbarn  # total pion cross section
    Kp = 0.5  # inelasticity
    B_hadr = (c * sigma_pp * Kp * rho / (mu * mp)).cgs  # hadronic loss coefficient
    # print(f"B_hadr = {B_hadr.cgs:.5e}")
    return B_hadr * ekin(p) * dp_dekin(p)


def b_coul(p, rho):
    """Coulomb loss coefficient, depends on normalised momentum.

    Dimensionless as it is normalised by time scaling factor tau0."""
    coul_coeff = -1e-18 * u.erg * u.cm**3 * u.s**-1
    n_e = 0.88 * rho / (mu * mp)
    B_coul = (coul_coeff * n_e / c / (mp * c)).cgs
    # print(f"B_coul = {B_coul.cgs:.5e}")
    return B_coul * (1 + p**-1.9)


# steady state spectrum with units of 1 / s
def f_p_0(alpha_inj, p_arr, rho):
    """Return the steady state spectrum for protons with given injection index."""
    return (
        (mp * c) ** (-alpha_inj + 2)  # scaling factor as we use normalised momentum
        # 1.0
        * p_arr ** (-alpha_inj + 1)
        / ((alpha_inj - 3) * (b_hadr(p_arr, rho) + b_coul(p_arr, rho)))
    )


# now calculate enratio for grid of alpha_inj
# alpha_inj_grid = np.linspace(4, 6, 30)
alpha_inj_grid = np.linspace(4, 5, 10)
dens_grid = np.logspace(-30, -18, 30) * u.g / u.cm**3

fp0_grid = np.zeros((len(alpha_inj_grid), len(dens_grid), len(pp_arr)))
log10_enpratio_grid = np.zeros((len(alpha_inj_grid), len(dens_grid)))

for iap, idens in itertools.product(
    range(len(alpha_inj_grid)),
    range(len(dens_grid)),
    desc="Computing enratio (proton): ",
    miniters=10,
    mininterval=1.0,
):
    # if iap > 0 or ib > 0:
    #     break
    alpha_inj = alpha_inj_grid[iap]
    dens = dens_grid[idens]
    fp0 = f_p_0(alpha_inj, pp_arr, dens)  # in units of (GeV/c)^(-a)
    # fp0 /= np.trapz(4 * np.pi * pp_arr**2 * fp0, x=pp_arr)
    ncr_p = np.trapz(4 * np.pi * pp_arr**2 * fp0, x=pp_arr)
    ecr_p = np.trapz(4 * np.pi * pp_arr**2 * ekin(pp_arr) * fp0, x=pp_arr)

    log10_enpratio_grid[iap, idens] = np.log10(
        (ecr_p * mp * c**2 / ncr_p).to_value(u.GeV)
    )
    fp0_grid[iap, idens, :] = fp0

print(1 / 10**log10_enpratio_grid.min(), 1 / 10**log10_enpratio_grid.max())

print("Calculation for proton grid done.")

# assert False

# plt.plot(alpha_inj_grid, log10_enpratio_grid)
# plt.savefig("./test.png")
# assert False

# now for electrons
print("Calculation for electron grid starting.")


def b_rad(p, B):
    """Loss coefficient for synchrotron & inverse compton effects."""
    sigma_thom = 0.665 * u.barn  # thomson cross section (from NIST)
    u_phot = (0.5 * u.eV / u.cm**3).to(
        u.erg / u.cm**3
    )  # photon energy density from ISRF, Mathis 1983
    u_mag = (B**2 / (2 * const.mu0)).to(
        u.erg / u.cm**3
    )  # magnetic field energy density
    B_rad = (4 / 3 * sigma_thom * c * (u_mag + u_phot) / (me * c**2)).cgs
    # print(f"B_rad = {B_rad.cgs:.5e}")
    return B_rad * p**2 * dp_dekin(p)


def f_e_0(alpha_inj_e, p_arr, B):
    """Return the steady state spectrum for electrons with given injection index."""
    return (
        (me * c) ** (-alpha_inj_e + 2) # scaling factor as we use normalised momentum
        * p_arr ** (-alpha_inj_e + 1) / ((alpha_inj_e - 3) * (b_rad(p_arr, B)))
    )


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


# now calculate enratio for grid of alpha_inj_e, magnetic field
# also for synchrotron frequencies
alpha_inj_e_grid = np.linspace(4, 5, 10)
magnetic_field_grid = np.logspace(-9, -3, 30) * u.G
synch_freq_grid = np.linspace(1.4, 12.4, 12) * u.GHz  # synchrotron frequency

fe0_grid = np.zeros(
    (
        len(alpha_inj_grid),
        len(dens_grid),
        len(alpha_inj_e_grid),
        len(magnetic_field_grid),
        len(pe_arr),
    )
)
log10_eneratio_grid = np.zeros(
    (
        len(alpha_inj_grid),
        len(dens_grid),
        len(alpha_inj_e_grid),
        len(magnetic_field_grid),
    )
)
nu_crit_grid = np.zeros((len(magnetic_field_grid), len(pe_arr)))
log10_ensynchratio_grid = np.zeros(
    (
        len(alpha_inj_grid),
        len(dens_grid),
        len(alpha_inj_e_grid),
        len(magnetic_field_grid),
        len(synch_freq_grid),
    )
)

for iap, idens, iae, ib in itertools.product(
    range(len(alpha_inj_grid)),
    range(len(dens_grid)),
    range(len(alpha_inj_e_grid)),
    range(len(magnetic_field_grid)),
    desc="Computing enratio (electron) & synchrotron ratio: ",
    miniters=100,
    mininterval=1.0,
):
    # if iap > 0 or ib > 0:
    #     break
    alpha_inj = alpha_inj_grid[iap]
    dens = dens_grid[idens]
    alpha_inj_e = alpha_inj_e_grid[iae]
    mag_field = magnetic_field_grid[ib]

    # print(alpha_inj, dens, alpha_inj_e, mag_field)

    # calculate the steady state spectra
    fp0 = f_p_0(alpha_inj, pp_arr, dens)
    fe0 = f_e_0(alpha_inj_e, pe_arr, mag_field)
    fe = (
        Kep
        * f_p_0(alpha_inj, pp0, dens)
        / f_e_0(alpha_inj_e, pe0, mag_field)
        * fe0
        * (me / mp).value
    )

    # calcualting en ratio with electrons
    ncr_p = np.trapz(4 * np.pi * pp_arr**2 * fp0, x=pp_arr)
    ecr_p = np.trapz(4 * np.pi * pp_arr**2 * ekin(pp_arr) * fp0, x=pp_arr) / ncr_p * mp * c**2
    ncr_e = np.trapz(4 * np.pi * pe_arr**2 * fe, x=pe_arr)
    ecr_e = np.trapz(4 * np.pi * pe_arr**2 * ekin(pe_arr) * fe, x=pe_arr) / ncr_e * me * c**2

    fe0_grid[iap, idens, iae, ib, :] = fe
    log10_eneratio_grid[iap, idens, iae, ib] = np.log10(
        (ecr_p / ecr_e).to_value(u.dimensionless_unscaled)
    )

    # now also calculate synchrotron emission
    nu_crit = crit_freq(mag_field, pe_arr)
    nu_crit_grid[ib, :] = nu_crit
    for isyn, synch_freq in enumerate(synch_freq_grid):
        synch_ker = synch_kernel(
            (synch_freq / nu_crit).to_value(u.dimensionless_unscaled)
        )

        # normalise synchrotron number density by total number of electrons
        ncr_esynch = np.trapz(
            y=4 * np.pi * pe_arr**2 * fe * synch_ker,
            x=pe_arr,
        ) / ncr_e

        log10_ensynchratio_grid[iap, idens, iae, ib, isyn] = np.log10(
            (ecr_p / ncr_esynch).to_value(u.GeV)
        )

    # print((ecr_esynch / ecr_p).to_value(u.dimensionless_unscaled), (ecr_e / ecr_p).to_value(u.dimensionless_unscaled))


print(
    1 / 10 ** log10_eneratio_grid.min(),
    1 / 10 ** log10_eneratio_grid.max(),
)
print(
    1 / 10 ** log10_ensynchratio_grid.min(),
    1 / 10 ** log10_ensynchratio_grid.max(),
)


# assert False
if os.path.exists(f"{data_dir}/steady_state_spectrum.h5"):
    os.remove(f"{data_dir}/steady_state_spectrum.h5")
with h5py.File(f"{data_dir}/steady_state_spectrum.h5", "a") as f:
    f.create_dataset("alpha_inj_grid", data=alpha_inj_grid)
    f.create_dataset("log10_dens_grid", data=np.log10(dens_grid.value))
    f.create_dataset("alpha_inj_e_grid", data=alpha_inj_e_grid)
    f.create_dataset("log10_B_grid", data=np.log10(magnetic_field_grid.value))
    f.create_dataset("syn_freq_grid", data=synch_freq_grid)

    prot_gr = f.create_group("proton")
    prot_gr.create_dataset("mom_grid", data=pp_arr)
    prot_gr.create_dataset("steady_state_grid", data=fp0_grid)
    prot_gr.create_dataset("log10_enpratio_grid", data=log10_enpratio_grid)

    prot_gr = f.create_group("electron")
    prot_gr.create_dataset("mom_grid", data=pe_arr)
    prot_gr.create_dataset("steady_state_grid", data=fe0_grid)
    prot_gr.create_dataset("log10_eneratio_grid", data=log10_eneratio_grid)
    prot_gr.create_dataset("log10_ensynchratio_grid", data=log10_ensynchratio_grid)
