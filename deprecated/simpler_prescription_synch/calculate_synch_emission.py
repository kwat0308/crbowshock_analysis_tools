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

args = parser.parse_args()

sim_name = args.sim_name
alpha_inj_e = float(args.alpha_inj_e)

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
        crit_freq_grid = np.clip(crit_freq_grid, 1e-5 * u.GHz, None)

        synch_kern = synch_coeff(Blos) * f_e[ie] * synch_kernel((synch_freq / crit_freq_grid).value)
        synch_emiss += synch_kern * dEe_arr[ie].to_value(u.GeV)
    
    return synch_emiss

def get_synch_lum(synch_args):
    """Calculate synchrotron luminosity for a given magnetic field and frequency grid."""
    mag_data, freq, dvol, je_data, ifreq = synch_args
    j_synch = synch_emissivity(mag_data, synch_freq=freq, alpha_e=alpha_inj_e)
    j_synch *= je_data
    L_synch = np.sum((j_synch * u.cm**-3).to_value(u.pc**-3) * dvol)
    del j_synch
    return ifreq, L_synch


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

# define synchrtron frequency grid
synch_freq_grid = np.logspace(-1, 1, 15) * u.GHz

# append specific synchrotron luminosity array
# synch_lum_arr = np.zeros((len(sim_files), len(synch_freq_grid))) * (u.erg / u.s / u.Hz)
# times = np.zeros(len(sim_files)) * u.s
synch_lum_arr = []
times = []

print(f"List of all simulation files: {sim_files}")

print("Before computation: ", psutil.Process(os.getpid()).memory_info().rss / 1024**2)

# get arguments for synchrotron luminosity computation
args_list = [(isim, sim_files[isim]) for isim in range(len(sim_files))]

for isim, sim_file in args_list:
    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

    with h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r") as pf:
        dens_data = pf["dens"][()][blist] * (u.g * u.cm**-3)
        encr_per_mass_data = pf["encr"][()][blist] * u.erg * u.g**-1
        magx_data = pf["magx"][()][blist] * (u.G)
        magy_data = pf["magy"][()][blist] * (u.G)
        magz_data = pf["magz"][()][blist] * (u.G)
        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]

        time = pf["real scalars"][()][0][1] * u.s # time in seconds

    low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

    if int(time.to_value(u.kyr)) > 180:
        print(f"Exceeded 180 kyr, stopping calculation")
        pf.close()
        break

    # synchrotron luminosity as in Longair eq 18.50
    encr_data = encr_per_mass_data * dens_data
    j_electron_data = (cre_prim_factor * encr_data + cre_second_factor * dens_data * encr_data * time) # should be in erg/cm^3

    print(np.min(j_electron_data), np.max(j_electron_data))

    # first calculate the magnetic field strength
    # assume isotropic magnetic field, so <B_perp> = sqrt(2/3) B
    mag_perp_data = np.sqrt(2/3) * np.sqrt(magx_data**2 + magy_data**2 + magz_data**2).to(u.uG)

    
    # mag_perp_data = np.sqrt(2/3) * mag_data

    del magx_data, magy_data, magz_data, encr_data, dens_data

    dvol_data = (
        np.tile(bsize[:, 0] / 8.0, reps=(8, 8, 8)).reshape((bsize.shape[0], 8, 8, 8))
        * u.cm
    ).to_value(u.pc)**3

    synch_args_list = [(mag_perp_data, freq, dvol_data, j_electron_data.value, ifreq) for ifreq, freq in enumerate(synch_freq_grid)]

    results = Parallel(n_jobs=8)(
        delayed(get_synch_lum)(synch_args) for synch_args in tqdm(synch_args_list, total=len(synch_args_list), desc=f"Computing synch luminosity for {sim_file}")
    )
    synch_lum_per_freq = np.zeros(len(synch_freq_grid))
    for ifreq, Lsyn in results:
        synch_lum_per_freq[ifreq] = Lsyn

    # synch_lum_per_freq = []

    # # for freq in tqdm(synch_freq_grid, total=len(synch_freq_grid), desc=f"Computing synch luminosity for {sim_file}"):
        # j_synch_normed = synch_emissivity(mag_perp_data, synch_freq=freq, alpha_e=alpha_inj_e)

        # j_synch_data = j_synch_normed * j_electron_data

        # synch_lum_per_freq.append(
            # np.sum((j_synch_data * u.cm**-3).to(u.pc**-3) * dvol_data) 
        # )

        # del j_synch_data, j_synch_normed
    


    synch_lum_arr.append(
        synch_lum_per_freq
    )

    times.append(time.to_value(u.kyr))
    print(f"Processed {sim_file}, time = {time.to(u.kyr)}")

    print("After computation: ", psutil.Process(os.getpid()).memory_info().rss / 1024**2)

synch_lum_arr = np.array(synch_lum_arr)
times = np.array(times)

# write data to h5py file
with h5py.File(f"{data_dir}/emission_data_synch_newversion_ae{alpha_inj_e}.h5", "a") as pf:
    sim_gr = pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]

    if "times" in pf[sim_name].keys():
        del pf[sim_name]["times"]
    sim_gr.create_dataset("times", data=times)

    if "synch_lum" in pf[sim_name].keys():
        del pf[sim_name]["synch_lum"]
    sim_gr.create_dataset("synch_lum", data=synch_lum_arr)

    if "synch_freq" in pf[sim_name].keys():
        del pf[sim_name]["synch_freq"]
    sim_gr.create_dataset("synch_freq", data=synch_freq_grid.to_value(u.GHz))


        

   