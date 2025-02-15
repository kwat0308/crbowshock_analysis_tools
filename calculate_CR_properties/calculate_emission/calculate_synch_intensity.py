import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import h5py
import os, psutil
import json
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from transform_data import get_transformed_data

import flash_amr_tools

import argparse

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs_new"
fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# load the bowshock config
bowshock_config = json.load(open("../../config/bowshock_config.json", "r"))["config"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--reset",
    "-r",
    action="store_true",
    default=False,
    help="Reset all synchrotron calculations or not.",
)

parser.add_argument(
    "--nprocs",
    action="store",
    default=8,
    help="Number of cores to use for parallelisation",
)

parser.add_argument(
    "--alpha_inj_e",
    "-ae",
    action="store",
    default=4.4,
    help="injection spectral index for electrons",
)

parser.add_argument(
    "--sim_name",
    "-S",
    action="store",
    type=str,
    help="name of sim",
    choices=list(bowshock_config.keys()),
)

args = parser.parse_args()

sim_name = args.sim_name

print(f"Performing calculation for {sim_name}")


with h5py.File(f"{data_dir}/steady_state_spectrum.h5", "r") as f:
    alpha_inj_grid = f["alpha_inj_grid"][()]
    alpha_inj_e_grid = f["alpha_inj_e_grid"][()]
    log10_B_grid = f["log10_B_grid"][()]
    synch_freq_grid = f["syn_freq_grid"][()] * u.GHz

    rho0 = f["proton/rho0"][()]

    log10_ensynchratio_grid = f["electron/log10_ensynchratio_grid"][()]


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

alpha_inj_e_idx = np.digitize(args.alpha_inj_e, alpha_inj_e_grid, right=True)
log10_ensynchratio_grid = log10_ensynchratio_grid[:, alpha_inj_e_idx, :, :]

# first interpolate the enratio grid
f_log10_ensynchratio = RegularGridInterpolator(
    (alpha_inj_grid, log10_B_grid),
    log10_ensynchratio_grid,
    method="linear",
    fill_value=None,
    bounds_error=False,
)

# get list of all files
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
    if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
        continue
    sim_files.append(file)

# sort
sim_files = sorted(sim_files)

if len(sim_files) > 200:
    sim_files = sim_files[::2]

# append specific synchrotron luminosity array
synch_lum_arr = np.zeros((len(sim_files), len(synch_freq_grid))) * (u.erg / u.s / u.Hz)
times = np.zeros(len(sim_files)) * u.s

print(f"List of all simulation files: {sim_files}")

print("Before computation: ", psutil.Process(os.getpid()).memory_info().rss / 1024**2)

# get arguments for synchrotron luminosity computation
args_list = [(isim, sim_files[isim]) for isim in range(len(sim_files))]

for arg in args_list:
    isim, sim_file = arg

    # loop over each file and get the slice
    blist, brefs, bns = flash_amr_tools.get_true_blocks(
        f"{flash_output_dir}/{sim_dir}/{sim_file}",
    )

    # remove units for memory efficiency
    pf = h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r")

    # LoS magnetic field (in Gauss)  ==  z-axis -> transverse is Bx and By contributions
    Blos_data = np.abs(pf["magx"][()][blist]) + np.abs(pf["magy"][()][blist])

    # get other properties from h5py file
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]
    time = pf["real scalars"][()][0][1] * u.s  # time in seconds (this is fine)

    low_cor, up_cor = bbox[0, :, 0], bbox[-1, :, 1]

    # # compute synchrotron luminosity
    # now we need to add astropy units since we would once convert to pc
    # as cm^3 with 1e21 overloads
    cell_size_data = (
        np.tile(bsize[:, 0] / 8.0, reps=(8, 8, 8)).reshape((bsize.shape[0], 8, 8, 8))
        * u.cm
    )

    for ifreq, synch_freq in tqdm(enumerate(synch_freq_grid), total=len(synch_freq_grid)):
        # encr to nele_synch ratio
        ensynchratio_data = (
            (
                10.0
                ** f_log10_ensynchratio(
                    (
                        # get_transformed_data(pf, "alpha_inj", blist),
                        4,
                        np.log10(Blos_data),
                    )
                )[..., ifreq]
            )
            * u.GeV
        ).to_value(u.erg)  # need to convert this to erg

        # multiply by density scaling factor (dimensionless since both in cgs)
        # needed since only numerator had dependence on density, which we didnt take into account
        ensynchratio_data /= (pf["dens"][()][blist] / rho0)

        # print(np.min(ensynchratio_data), np.max(ensynchratio_data))

        # specific synchrotron intensity is in -> E dN/(dV dt), in units of erg / s / Hz / cm^3
        # NB: astropy doesnt properly take care of cgs units with magnetic fields, so we convert them to values instead and add units later
        synch_coeff = (
            np.sqrt(3)
            * Blos_data  # LoS B-field in Gauss
            * const.e.gauss.value**3
            / (const.m_e.cgs.value * const.c.cgs.value**2)
        )

        # synchrotron intensity in  (u.erg * u.s**-1 * u.Hz**-1 * u.cm**-3)
        synch_intens_data = (
            synch_coeff
            * pf["encr"][()][blist]  # specific CR energy
            * pf["dens"][()][blist]  # density
        ) / ensynchratio_data

        if isim % 5 == 0:
            synch_idx = np.digitize(1.4 * u.GHz, synch_freq_grid, right=True)
            if ifreq != synch_idx:
                continue
            synch_intens_sl = flash_amr_tools.get_slice(
                synch_intens_data * synch_freq.to_value(u.Hz),
                pos=0.0,
                axis=2,
                ref_lvl=ref_lvl,
                bbox=bbox,
                bsize=bsize,
                brefs=brefs,
                bns=bns,
            )
            # write data to h5py file
            with h5py.File(f"{data_dir}/emission_data_synch.h5", "a") as pfo:
                sim_gr = (
                    pfo.create_group(sim_name)
                    if sim_name not in pfo.keys()
                    else pfo[sim_name]
                )

                if "slices" in sim_gr.keys():
                    del pfo[sim_name]["slices"]

                synch_sl_dr = sim_gr.create_group("slices")

                synch_sl_dr.create_dataset(f"t{isim}0kyr", data=synch_intens_sl)
                synch_sl_dr.create_dataset("bbox", data=bbox)

            del synch_intens_sl

        synch_lum_arr[isim, ifreq] = np.sum(
            (synch_intens_data * (u.erg / u.s / u.Hz / u.cm**3)).to(
                u.erg / u.s / u.Hz / u.pc**3
            )
            * cell_size_data.to(u.pc) ** 3,
            axis=(0, 1, 2, 3),
        )

    # del synch_intens_data
    del cell_size_data
    del synch_coeff
    del Blos_data

    times[isim] = time

    pf.close()

    print(
        "Memory Usage: ",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        " MiB",
    )

print(synch_lum_arr.min(), synch_lum_arr.max())

# sort arrays based on increasing orde rin time
synch_lum_arr = synch_lum_arr[np.argsort(times)]
times = np.sort(times)

# write data to h5py file
with h5py.File(f"{data_dir}/emission_data_synch.h5", "a") as pf:
    sim_gr = pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]

    if "times" in pf[sim_name].keys():
        del pf[sim_name]["times"]
    sim_gr.create_dataset("times", data=times.to(u.kyr))

    if "synch_lum" in pf[sim_name].keys():
        del pf[sim_name]["synch_lum"]
    sim_gr.create_dataset("synch_lum", data=synch_lum_arr.to(u.erg / u.s / u.Hz))

    if "synch_freq" in pf[sim_name].keys():
        del pf[sim_name]["synch_freq"]
    sim_gr.create_dataset("synch_freq", data=synch_freq_grid.to_value(u.GHz))
