import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import h5py

data_dir = "/home/watanabe/Projects/meerkat_bowshocks/meerkat_nrfs/meerkat_nrfs/data/"

# constants used
mp = const.m_p.cgs
me = const.m_e.cgs
c = const.c.cgs
synch_freq = 1.28 * u.GHz   # synchrotron frequency from telescope

def synch_kernel(x):
    '''As in Aharonian et al. 2010'''
    exp_factor = (1 + 0.884 * x**(2/3) + 0.471 * x**(4/3)) / (1 + 1.64 * x**(2/3) + 0.974 * x**(4/3))
    return 2.15 * x**(1/3) * (1 + 3.06 * x)**(1/6) * exp_factor * np.exp(-x)

def crit_freq(Blos, pele):
    '''Critical frequency as a function of magnetic field / microG and electron energy / GeV'''
    return 0.5 * (Blos / (1 * u.uG)) * (np.sqrt(1 + (pele / (me * c))**2) / 1e4)**2 * u.GHz

def ekin(p, m=const.m_p.cgs):
    '''kinetic energy of proton in GeV'''
    return (np.sqrt(p**2*c**2 + mp**2*c**4) - mp*c**2).to(u.GeV)

def pele(pprot):
    '''secondary electron momentum from pion loss as function of proton momentum'''
    return (np.sqrt((ekin(pprot) / 16 + me*c**2)**2 - me**2*c**4) / c).to(u.cm * u.g / u.s)

with h5py.File(f"{data_dir}/steady_state_spectrum_thr.h5", "r") as f:
    log10_R0_grid = f["log10_R0_grid"][()]
    log10_vel_grid = f["log10_vel_grid"][()]
    log10_kappa10_grid = f["log10_kappa10_grid"][()]
    log10_dens_grid = f["log10_dens_grid"][()]
    alpha_inj_grid = f["alpha_inj_grid"][()]
    log10_pprot_grid = f["log10_pprot_grid"][()]
    steady_state_mom_grid = f["steady_spect_mom_grid"][()]

pprot_grid = (10.0**log10_pprot_grid * (u.cm * u.g / u.s))
pprot_grid_calc = pprot_grid[:,None,None,None,None,None]
pele_grid = pele(pprot_grid)
pele_grid_calc = pele(pprot_grid_calc)

# LoS magnetic field
Blos_interp_grid = np.logspace(-10, -4, 100) * u.G

log10_ensynch_grid = np.zeros((len(alpha_inj_grid), len(log10_vel_grid), len(log10_kappa10_grid), len(log10_R0_grid), len(log10_dens_grid), len(Blos_interp_grid)))

encr_grid = np.trapz(y=4 * np.pi * pprot_grid_calc**2 * ekin(pprot_grid_calc) * steady_state_mom_grid, x=pprot_grid, axis=0)

for ib, Blos in enumerate(Blos_interp_grid):
    nuc = crit_freq(Blos, pele_grid)
    synch_ker = synch_kernel((synch_freq / nuc).to_value(u.dimensionless_unscaled))

    nelesynch = np.trapz(y=4 * np.pi * pele_grid_calc**2 * steady_state_mom_grid * synch_ker[:,None,None,None,None,None], x=pele_grid, axis=0)
    
    log10_ensynch_grid[...,ib] = np.log10((encr_grid / nelesynch).to_value(u.GeV))

with h5py.File(f"{data_dir}/steady_state_spectrum_thr.h5", "a") as f:
    synch = f.create_group("synch_emission")
    synch.create_dataset("log10_Blos_grid", data=np.log10(Blos_interp_grid.to_value(u.G)))
    synch.create_dataset("synch_freq", data=synch_freq.to_value(u.GHz))
    synch.create_dataset("log10_ensynch_grid", data=log10_ensynch_grid)
