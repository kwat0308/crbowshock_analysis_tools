import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const

from scipy.integrate import quad
from tqdm import tqdm
from tqdm.contrib import itertools

from joblib import delayed, Parallel

import h5py

data_dir = "../../data"


def inj_spect(ep, alpha_inj):
    """Unnormalised injection spectrum, unitless"""
    return (ep**2 - 1) ** (-0.5 * (alpha_inj + 1)) * ep


def mom(e):
    """momentum as a function of normalised energy, in cgs units"""
    return mp * c * np.sqrt(e**2 - 1)


# energy parameters
mp = const.m_p.cgs
me = const.m_e.cgs
c = const.c.cgs

# diffusion
delta = 0.5
E0 = 10 * u.GeV
e0 = (E0 / (mp * c**2)).to_value(u.dimensionless_unscaled)

# hadronic loss parameters
sigma_pp = 44 * u.mbarn  # total pion cross section
Kp = 0.5  # inelasticity
mol_weight = 1.4  # mean molecular weight
B_hadr = (c * sigma_pp * Kp / (mp * mol_weight)).cgs  # hadronic loss coefficient
print(f"B_hadr = {B_hadr.cgs:.5e}")

# grid of energies from 1.3 GeV to 10 TeV since we want > pion threshold
Ekin_arr = np.logspace(np.log10(1.3), 4, 200) * u.GeV
ekin_arr = (Ekin_arr / (mp * c**2)).to_value(u.dimensionless_unscaled)
print("Minimum & maximum of E / mc^2: ", ekin_arr.min(), ekin_arr.max())

# grid parameters
R0_min = 1e-3 * u.pc
R0_max = 5 * u.pc
vmin = 1e-2 * (u.km / u.s)
vmax = 5e3 * (u.km / u.s)
kappa10_min = 3e22 * (u.cm**2 / u.s)
kappa10_max = 3e29 * (u.cm**2 / u.s)
dens_min = 1e-30 * (u.g / u.cm**3)
dens_max = 1e-18 * (u.g / u.cm**3)

vel_grid = np.logspace(np.log10(vmin.value), np.log10(vmax.value), 25) * (u.km / u.s)
R0_grid = np.logspace(np.log10(R0_min.value), np.log10(R0_max.value), 25) * u.pc
kappa_10_grid = np.logspace(
    np.log10(kappa10_min.value), np.log10(kappa10_max.value), 25
) * (u.cm**2 / u.s)
dens_grid = np.logspace(np.log10(dens_min.value), np.log10(dens_max.value), 25) * (
    u.g / u.cm**3
)
alpha_inj_grid = np.linspace(4, 10, 25)

# now compute steady state grid
steady_state_grid = np.zeros(
    (
        len(ekin_arr),
        len(alpha_inj_grid),
        len(kappa_10_grid),
        len(R0_grid),
        len(vel_grid),
        len(dens_grid),
    )
)


def get_steady_state(args):
    """Wrapper function to compute steady state spectrum"""
    alpha_inj, ekin_arr, kappa_10_grid, R0_grid, vel_grid, dens_grid = args

    inj_spectrum_grid = inj_spect(ekin_arr, alpha_inj)
    steady_state_grid_tmp = np.zeros(
        (len(ekin_arr), len(kappa_10_grid), len(R0_grid), len(vel_grid), len(dens_grid))
    )

    for ik, ir, iv, id in itertools.product(
        range(len(kappa_10_grid)),
        range(len(R0_grid)),
        range(len(vel_grid)),
        range(len(dens_grid)),
        desc="Computing Steady State Spectrum",
        miniters=1000,
        mininterval=1.0,
    ):
        R0 = R0_grid[ir]
        kappa_10 = kappa_10_grid[ik]
        vel = vel_grid[iv]
        rho = dens_grid[id]

        tau_adv = (R0 / vel).to_value(u.s)
        taudiff_coeff = (R0**2 / kappa_10 * e0**delta).to_value(u.s)
        bhadr_coeff = (B_hadr * rho).to_value(u.s**-1)

        # analytical function
        for ie, e in enumerate(ekin_arr):
            green_int = bhadr_coeff**-1 * (
                tau_adv**-1 * np.log(ekin_arr / e)
                + (taudiff_coeff**-1 / delta) * (ekin_arr**delta - e**delta)
            )
            steady_state_grid_tmp[ie, ik, ir, iv, id] = np.trapz(
                inj_spectrum_grid[ie:, ...]
                * (bhadr_coeff * ekin_arr[ie:]) ** -1
                * np.exp(-green_int[ie:]),
                x=ekin_arr[ie:],
                axis=0,
            )

    return steady_state_grid_tmp


# construct arguments
args = [
    (alpha_inj, ekin_arr, kappa_10_grid, R0_grid, vel_grid, dens_grid)
    for alpha_inj in alpha_inj_grid
]

# parallelize
results = Parallel(n_jobs=16)(delayed(get_steady_state)(arg) for arg in args)

for ia, steady_state_tmp in enumerate(results):
    steady_state_grid[:, ia, ...] = steady_state_tmp


pprot_grid = mom(ekin_arr)
pprot_grid_calc = pprot_grid[:, None, None, None, None, None]
ekin_arr_calc = ekin_arr[:, None, None, None, None, None]

steady_state_mom_grid = (
    steady_state_grid * pprot_grid_calc / (2 * mp**2 * c**2 * ekin_arr_calc)
)  # for the factor of dE / dp
ncr_grid = np.trapz(
    y=4 * np.pi * pprot_grid_calc**2 * steady_state_mom_grid, x=pprot_grid, axis=0
)
encr_grid = np.trapz(
    y=4 * np.pi * pprot_grid_calc**2 * ekin_arr_calc * steady_state_mom_grid,
    x=pprot_grid,
    axis=0,
)

log10_enratio_grid = np.log10(((encr_grid * mp * c**2) / ncr_grid).to_value(u.GeV))

# save in h5py file
with h5py.File(f"{data_dir}/steady_state_spectrum.h5", "w") as f:
    f.create_dataset("alpha_inj_grid", data=alpha_inj_grid)
    f.create_dataset("log10_R0_grid", data=np.log10(R0_grid.value))
    f.create_dataset("log10_vel_grid", data=np.log10(vel_grid.value))
    f.create_dataset("log10_kappa10_grid", data=np.log10(kappa_10_grid.value))
    f.create_dataset("log10_dens_grid", data=np.log10(dens_grid.value))
    f.create_dataset("log10_pprot_grid", data=np.log10(mom(ekin_arr).value))
    f.create_dataset("steady_spect_grid", data=steady_state_grid)
    f.create_dataset("steady_spect_mom_grid", data=steady_state_mom_grid)
    f.create_dataset("log10_enratio_grid", data=log10_enratio_grid)
