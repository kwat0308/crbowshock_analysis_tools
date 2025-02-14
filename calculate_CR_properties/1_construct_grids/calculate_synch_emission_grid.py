import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import h5py
from tqdm import tqdm

data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# constants used
mp = const.m_p.cgs
me = const.m_e.cgs
c = const.c.cgs


def synch_kernel(x):
    """As in Aharonian et al. 2010"""
    exp_factor = (1 + 0.884 * x ** (2 / 3) + 0.471 * x ** (4 / 3)) / (
        1 + 1.64 * x ** (2 / 3) + 0.974 * x ** (4 / 3)
    )
    return 2.15 * x ** (1 / 3) * (1 + 3.06 * x) ** (1 / 6) * exp_factor * np.exp(-x)


def crit_freq(Blos, pele):
    """Critical frequency as a function of magnetic field / microG and electron energy / GeV"""
    # NB: here we take normalised momentum (normalised over mass)
    return (
        0.5
        * (Blos / (1 * u.uG))
        * (np.sqrt(1 + pele ** 2) / 1e4) ** 2
        * u.GHz
    )


# def pele(pprot):
#     '''secondary electron momentum from pion loss as function of proton momentum'''
#     return (np.sqrt((ekin(pprot) / 16 + me*c**2)**2 - me**2*c**4) / c).to(u.cm * u.g / u.s)

with h5py.File(f"{data_dir}/steady_state_spectrum.h5", "r") as f:
    log10_R0_grid = f["log10_R0_grid"][()]
    log10_vel_grid = f["log10_vel_grid"][()]
    log10_kappa10_grid = f["log10_kappa10_grid"][()]
    log10_dens_grid = f["log10_dens_grid"][()]
    alpha_inj_grid = f["alpha_inj_grid"][()]

    pprot_grid = 10.0**f["prot/log10_mom_grid"][()]  # unitless, normalised to proton mass
    pprot_grid_calc = pprot_grid[:, None, None, None, None, None]
    ekin_arr_calc = np.sqrt(pprot_grid_calc**2 + 1) - 1

    # spectrum is normalised so here grid will get back Ecr in normalised units -> need to 
    # multiply by mp c^2
    encr_grid = (np.trapz(
        y=4 * np.pi * pprot_grid_calc**2 * ekin_arr_calc * f["prot/steady_spect_mom_grid"][()],
        x=pprot_grid,
        axis=0,
    )  * mp * c**2).to(u.GeV)

    del pprot_grid
    del pprot_grid_calc
    del ekin_arr_calc

    # same here, use normalised momentumn & normalised steady state spectrum
    pele_grid = 10.0**f["ele/log10_mom_grid"][()] 
    pele_grid_calc = pele_grid[:, None, None, None, None, None]
    f_ele_grid = f["ele/steady_spect_mom_grid"][()]

# LoS magnetic field
Blos_interp_grid = np.logspace(-7, -4, 30) * u.G

# synchrotron frequency
synch_freq_grid = np.linspace(1.4, 12.4, 12) * u.GHz

log10_ensynch_grid = np.zeros(
    (
        len(alpha_inj_grid),
        len(log10_kappa10_grid),
        len(log10_R0_grid),
        len(log10_vel_grid),
        len(log10_dens_grid),
        len(Blos_interp_grid),
        len(synch_freq_grid),
    )
)

for ib, Blos in enumerate(
    tqdm(
        Blos_interp_grid,
        desc="Computing for each B-field: ",
    )
):

    nuc = crit_freq(Blos, pele_grid).to(u.GHz)
    for isyn, synch_freq in enumerate(synch_freq_grid):

        synch_ker = synch_kernel((synch_freq / nuc).to_value(u.dimensionless_unscaled))[
            :, None, None, None, None, None
        ]
        
        # number of electrons contributing to synchrotron density
        # this is unitless since everything here is unitless
        nelesynch = np.trapz(
            y=4 * np.pi * pele_grid_calc**2 * f_ele_grid * synch_ker,
            x=pele_grid,
            axis=0,
        )

        # divide CR energy from protons with number of electrons
        log10_ensynch_grid[..., ib, isyn] = np.log10(
            (encr_grid / nelesynch).to_value(u.GeV)
        )

print(np.min(log10_ensynch_grid), np.max(log10_ensynch_grid))

with h5py.File(f"{data_dir}/steady_state_spectrum.h5", "a") as f:
    if "synch_emission" in list(f.keys()):
        del f["synch_emission"]
    synch = f.create_group("synch_emission")
    synch.create_dataset(
        "log10_Blos_grid", data=np.log10(Blos_interp_grid.to_value(u.G))
    )
    synch.create_dataset("synch_freq_grid", data=synch_freq_grid.to_value(u.GHz))
    synch.create_dataset("log10_ensynch_grid", data=log10_ensynch_grid)
