import numpy as np
import os, psutil
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const

from scipy.integrate import quad
from tqdm import tqdm
from tqdm.contrib import itertools

from joblib import delayed, Parallel

import h5py

import json
import flash_amr_tools

import argparse

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs_new"
fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# load the bowshock config
bowshock_config = json.load(open("../../config/bowshock_config.json", "r"))["config"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--sim_name",
    "-S",
    action="store",
    type=str,
    help="name of sim",
    choices=list(bowshock_config.keys()),
)

parser.add_argument(
    "--alpha_inj_e",
    "-ae",
    action="store",
    default=2.2,
    help="injection spectral index for electrons",
    type=float,
)

parser.add_argument(
    "--synch_freq",
    "-f",
    action="store",
    default=1.0,
    help="synchrotron frequency in GHz",
    type=float,
)

args = parser.parse_args()

sim_name = args.sim_name
alpha_inj_e = float(args.alpha_inj_e)
synch_freq = args.synch_freq * u.GHz

# energy parameters
mp = const.m_p.cgs
me = const.m_e.cgs
c = const.c.cgs
mu = 0.61  # mean molecular weight

Ee_arr_binedges = np.logspace(0, 4, 51) * u.GeV  # kinetic energy
Ee_arr = np.sqrt(Ee_arr_binedges[1:] * Ee_arr_binedges[:-1])  # geometric mean
dEe_arr = (Ee_arr_binedges[1:] - Ee_arr_binedges[:-1]).to(u.GeV)

def mom(Ek, mass=me):
    """Calculate momentum from kinetic energy."""
    return (np.sqrt((Ek + mass * c**2) ** 2 - (mass * c**2) ** 2) / c).to(u.g * u.cm / u.s)

def spect(Es, alpha):
    """Power-law spectrum."""
    f_e = Es**(-alpha) * 4 * np.pi * mom(Es)**2 * (2 * mom(Es) * c**2) / (Es + me * c**2)
    f_e /= np.trapz(f_e, Es, axis=0)
    return f_e.value

# below are functions to calculation synchrotron emitting electrons
def synch_kernel(x):
    """Synchrotron kernal, as in Aharonian et al. 2010."""
    exp_factor = (1 + 0.884 * x ** (2 / 3) + 0.471 * x ** (4 / 3)) / (
        1 + 1.64 * x ** (2 / 3) + 0.974 * x ** (4 / 3)
    )
    return 2.15 * x ** (1 / 3) * (1 + 3.06 * x) ** (1 / 6) * exp_factor * np.exp(-x)


def crit_freq(Blos, eele):
    """Critical frequency as a function of magnetic field / microG and electron energy / GeV"""
    # NB: here we take normalised momentum (normalised over mass)
    return 0.5 * (Blos / (1 * u.uG)) * (np.sqrt(1 + (mom(eele) / (me * c))**2) / 1e4) ** 2 * u.GHz

def synch_coeff(Blos):
    """Coefficient in front of synchrotron integral."""
    return (np.sqrt(3) * const.e.gauss.value**3 * Blos.to_value(u.G)) / (me.value * c.value**2)

def synch_emissivity(Blos, synch_freq=1*u.GHz, alpha_e=2.7):
    """Calculate synchrotron emissivity for a given magnetic field."""
    synch_emiss = np.zeros_like(Blos.value)
    f_e = spect(Ee_arr, alpha_e)

    for ie, eele in enumerate(Ee_arr):
        crit_freq_grid = crit_freq(Blos.to(u.uG), eele)

        synch_kern = synch_coeff(Blos) * f_e[ie] * synch_kernel((synch_freq / crit_freq_grid).value)
        synch_emiss += synch_kern * dEe_arr[ie].to_value(u.GeV)
    
    return synch_emiss


if not os.path.exists(f"{fig_dir}/synch"):
    os.mkdir(f"{fig_dir}/synch")

if not os.path.exists(f"{fig_dir}/synch/{sim_name}"):
    os.mkdir(f"{fig_dir}/synch/{sim_name}")

# get list of all files
sim_dir = bowshock_config[sim_name]["sim_dir"]
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
    if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
        continue
    sim_files.append(file)

# sort
sim_files = sorted(sim_files)

# set the primary electron fraction to be ~ 0.01
cre_prim_factor = 0.01

sigma_pp = 44 * u.mbarn  # total pion cross section
Kp = 0.5  # inelasticity
mol_weight = 1.4  # mean molecular weight
Bhadr_coeff = (const.c.cgs * sigma_pp * Kp / (const.m_p.cgs * mol_weight)).cgs  # hadronic loss coefficient

cre_second_factor = 1/6 * Bhadr_coeff

times_to_store = [30, 90, 120, 180] # in kyr

print(f"List of all simulation files: {sim_files}")

print("Before computation: ", psutil.Process(os.getpid()).memory_info().rss / 1024**2)

# get arguments for synchrotron luminosity computation
args_list = [(isim, sim_files[isim]) for isim in range(len(sim_files))]

for isim, sim_file in args_list:
    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

    with h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r") as pf:
        dens_data = pf["dens"][()][blist]  * (u.g / u.cm**3)
        encr_per_mass_data = pf["encr"][()][blist] * (u.erg / u.g)
        magx_data = pf["magx"][()][blist] * (u.G)
        magy_data = pf["magy"][()][blist] * (u.G)
        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]

        time = pf["real scalars"][()][0][1] * u.s # time in seconds

    low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

    if int(time.to_value(u.kyr)) not in times_to_store:
        pf.close()
        continue

    print(f"Processing file {sim_file} at time {time.to_value(u.kyr):.1f} kyr")

    # synchrotron luminosity as in Longair eq 18.50
    encr_data = encr_per_mass_data * dens_data
    j_electron_data = (cre_prim_factor * encr_data + cre_second_factor * dens_data * encr_data * time) # should be in erg/cm^3

    print(np.min(j_electron_data), np.max(j_electron_data))

    # first calculate the magnetic field strength
    mag_data = magx_data**2 + magy_data**2
    mag_data = np.sqrt(mag_data).to(u.uG)

    # assume isotropic magnetic field, so <B_perp> = sqrt(2/3) B
    mag_perp_data = np.sqrt(2/3) * mag_data

    del mag_data, magx_data, magy_data, encr_data, dens_data, encr_per_mass_data

    j_synch_data = synch_emissivity(mag_perp_data, synch_freq=synch_freq, alpha_e=alpha_inj_e)
    j_synch_data *= j_electron_data.value

    j_synch_sl = flash_amr_tools.get_slice(j_synch_data, pos=0.0, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
    # write data to h5py file
    with h5py.File(
        f"{data_dir}/emission_data_synch_newversion_ae{alpha_inj_e}.h5", "a"
    ) as pfo:

        sim_gr = (
            pfo.create_group(sim_name)
            if sim_name not in pfo.keys()
            else pfo[sim_name]
        )

        synch_sl_dr = (
            sim_gr["slices"]
            if "slices" in sim_gr.keys()
            else sim_gr.create_group("slices")
        )

        if f"t{time.to_value(u.kyr):.0f}kyr" in synch_sl_dr.keys():
            del synch_sl_dr[f"t{time.to_value(u.kyr):.0f}kyr"]
        if f"t{time.to_value(u.kyr):.0f}_bbox" in synch_sl_dr.keys():
            del synch_sl_dr[f"t{time.to_value(u.kyr):.0f}_bbox"]

        synch_sl_dr.create_dataset(f"t{time.to_value(u.kyr):.0f}kyr", data=j_synch_sl)
        synch_sl_dr.create_dataset(f"t{time.to_value(u.kyr):.0f}_bbox", data=bbox)

    del j_synch_sl, j_synch_data, j_electron_data, mag_perp_data

    pf.close()

    print("After computation: ", psutil.Process(os.getpid()).memory_info().rss / 1024**2)


        

   