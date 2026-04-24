import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import h5py
import os, psutil
import json
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from joblib import Parallel, delayed

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from transform_data import get_transformed_data
from synch_emission_functions import *

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
    "--alpha_inj_e",
    "-ae",
    action="store",
    default=4.2,
    help="injection spectral index for electrons",
)

parser.add_argument(
    "--alpha_inj_p",
    "-ap",
    action="store",
    default=4.2,
    help="injection spectral index for proton",
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

# here we define the normalised momentum grid for both protons and electrons,
p_arr = np.logspace(-3, 6, 200)  # in units of m c
Kep, pp0, pe0 = (
    ep_ratio()
)  # get the normalisation for the electron spectrum based on the proton spectrum at 10 GeV, and the corresponding momenta

synch_freqs = np.linspace(0.5, 15, 30) * u.GHz  # in Hz

sim_name = args.sim_name

print(f"Performing calculation for {sim_name}")


# if not os.path.exists(f"{fig_dir}/synch"):
#     os.mkdir(f"{fig_dir}/synch")

# if not os.path.exists(f"{fig_dir}/synch/{sim_name}"):
#     os.mkdir(f"{fig_dir}/synch/{sim_name}")


# get list of all files
sim_dir = bowshock_config[sim_name]["sim_dir"]
output_dir = f"{flash_output_dir}/{sim_dir}"
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
    if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
        continue
    sim_files.append(file)

# sort
sim_files = sorted(sim_files)

# get list of all files
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
    if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
        continue
    sim_files.append(file)

# sort
sim_files = sorted(sim_files)

# append specific synchrotron luminosity array
# synch_lum_arr = np.zeros((len(sim_files), len(synch_freq_grid))) * (u.erg / u.s / u.Hz)
# times = np.zeros(len(sim_files)) * u.s
synch_lum_arr = []
times = []

print(f"List of all simulation files: {sim_files}")

print("Before computation: ", psutil.Process(os.getpid()).memory_info().rss / 1024**2)

# get arguments for synchrotron luminosity computation
args_list = [(isim, sim_files[isim]) for isim in range(len(sim_files))]

for arg in args_list:
    isim, sim_file = arg
    # remove units for memory efficiency
    pf = h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r")
    time = pf["real scalars"][()][0][1] * u.s  # time in seconds (this is fine)

    if time in times:
        print(f"Time {time.to(u.kyr)} already computed, skipping")
        pf.close()
        continue

    print(f"Processing {sim_file}, time = {time.to(u.kyr)}")

    # loop over each file and get the slice
    blist, brefs, bns = flash_amr_tools.get_true_blocks(
        f"{flash_output_dir}/{sim_dir}/{sim_file}",
    )

    # LoS magnetic field (in Gauss)  ==  z-axis -> transverse is Bx and By contributions
    # Blos_data = np.abs(pf["magx"][()][blist]) + np.abs(pf["magy"][()][blist])
    Blos_data = (2 / 3) * np.sqrt(
        pf["magx"][()][blist] ** 2
        + pf["magy"][()][blist] ** 2
        + pf["magz"][()][blist] ** 2
    )
    Blos_data[Blos_data < 1e-30] = 1e-30  # set a floor to avoid issues
    dens_data = pf["dens"][()][blist]
    dens_data[dens_data < 1e-40] = 1e-40  # set a floor to avoid issues
    vel_data = np.sqrt(
        pf["velx"][()][blist] ** 2
        + pf["vely"][()][blist] ** 2
        + pf["velz"][()][blist] ** 2
    )
    encr_data = (
        (pf["encr"][()][blist] * pf["dens"][()][blist]) * u.erg / u.cm**3
    )  #  CR energy density in erg/cm^3

    synch_output_file = f"{output_dir}/emission_data_t{time.to(u.kyr).value:.0f}.h5"
    synch_output_h5 = h5py.File(synch_output_file, "w")

    # calculate the energy loss terms for each
    b_hadr_data = b_hadr(p_arr[:, None, None, None], dens_data[None, :, :, :])
    synch_output_h5.create_dataset(
        "b_hadr", data=b_hadr_data, compression="gzip", compression_opts=9
    )
    del b_hadr_data
    b_coul_data = b_coul(p_arr[:, None, None, None], dens_data[None, :, :, :])
    synch_output_h5.create_dataset(
        "b_coul", data=b_coul_data, compression="gzip", compression_opts=9
    )
    del b_coul_data
    b_adiab_data = b_adiab(p_arr[:, None, None, None], vel_data[None, :, :, :])
    synch_output_h5.create_dataset(
        "b_adiab", data=b_adiab_data, compression="gzip", compression_opts=9
    )
    del b_adiab_data
    b_synch_data = b_synch(p_arr[:, None, None, None], Blos_data[None, :, :, :])
    synch_output_h5.create_dataset(
        "b_synch", data=b_synch_data, compression="gzip", compression_opts=9
    )
    del b_synch_data
    b_ic_photon_data = b_ic_photon(p_arr[:, None, None, None])
    synch_output_h5.create_dataset(
        "b_ic_photon", data=b_ic_photon_data, compression="gzip", compression_opts=9
    )
    del b_ic_photon_data
    b_ic_star_data = b_ic_star(p_arr[:, None, None, None], Tstar=1e4 * u.K)
    synch_output_h5.create_dataset(
        "b_ic_star", data=b_ic_star_data, compression="gzip", compression_opts=9
    )
    del b_ic_star_data

    # similarly calculate the steady state spectra for protons and electrons
    f_p_0_data = f_p_0(
        args.alpha_inj_p,
        p_arr[:, None, None, None],
        dens_data[None, :, :, :],
        vel_data[None, :, :, :],
        pmax=bowshock_config[sim_name]["pmax_prot"],
    )

    # number density also
    # this is calcualted by the ratio of n / ecr * ecr_data, where n is the number density of CRs calculated by integrating over the momentum grid, and ecr_data is the CR energy density from the simulation data

    ncr_p = np.trapz(
        4 * np.pi * p_arr[:, None, None, None] ** 2 * f_p_0_data,
        x=p_arr[:, None, None, None],
        axis=0,
    )
    ecr_p = np.trapz(
        4 * np.pi * p_arr[:, None, None, None] ** 2 * ekin(p_arr[:, None, None, None]) * f_p_0_data,
        x=p_arr[:, None, None, None],
        axis=0,
    ) / ncr_p * (mpc2)

    ncr_p_data = (ncr_p / ecr_p * encr_data).to_value(u.cm**-3)
    synch_output_h5.create_dataset(
        "ncr_p", data=ncr_p_data, compression="gzip", compression_opts=9
    )
    del ncr_p_data

    synch_output_h5.create_dataset(
        "f_p_0", data=f_p_0_data, compression="gzip", compression_opts=9
    )
    del f_p_0_data

    f_e_0_data = f_e_0(
        args.alpha_inj_e,
        p_arr[:, None, None, None],
        Blos_data[None, :, :, :],
        Tstar=1e4 * u.K,  # assumed based on stellar track from Ekstrom+2012
        vel=vel_data[None, :, :, :],
        pmax=bowshock_config[sim_name]["pmax_elec"],
    )

    # for electrons, need to renormalise the spectrum based on the ep ratio at 10 GeV
    f_e_data = (
        Kep
        * f_p_0(
            args.alpha_inj_p,
            pp0,
            dens_data[:, :, :],
            vel_data[:, :, :],
            pmax=bowshock_config[sim_name]["pmax_prot"],
        )
        / f_e_0(
            args.alpha_inj_e,
            pe0,
            Blos_data[:, :, :],
            Tstar=1e4 * u.K,
            vel=vel_data[:, :, :],
            pmax=bowshock_config[sim_name]["pmax_elec"],
        )
        * f_e_0_data
    )
    synch_output_h5.create_dataset(
        "f_e_0", data=f_e_data, compression="gzip", compression_opts=9
    )
    del f_e_0_data

    ncr_e = np.trapz(
        4 * np.pi * p_arr[:, None, None, None] ** 2 * f_e_data,
        x=p_arr[:, None, None, None],
        axis=0,
    )
    ecr_e = np.trapz(
        4 * np.pi * p_arr[:, None, None, None] ** 2 * ekin(p_arr[:, None, None, None]) * f_e_data,
        x=p_arr[:, None, None, None],
        axis=0,
    ) / ncr_e * (mec2)

    ncr_e_data = (ncr_e / ecr_p * encr_data).to_value(u.cm**-3)
    encr_e_data = (ecr_e / ecr_p * encr_data).to_value(u.erg / u.cm**3)
    synch_output_h5.create_dataset(
        "ncr_e", data=ncr_e_data, compression="gzip", compression_opts=9
    )
    synch_output_h5.create_dataset(
        "encr_e", data=encr_e_data, compression="gzip", compression_opts=9
    )
    del ncr_e_data
    del encr_e_data
    del ecr_e

    # now for synchrotron emission related stuff
    nu_crit_data = crit_freq(Blos=Blos_data[None, :, :, :], pele=p_arr[:, None, None, None])
    synch_output_h5.create_dataset(
        "nu_crit", data=nu_crit_data, compression="gzip", compression_opts=9
    )

    synch_coeff = (
        np.sqrt(3)
        * Blos_data  # LoS B-field in Gauss
        * const.e.gauss.value**3
        / (const.m_e.cgs.value * const.c.cgs.value**2)
    )
    synch_output_h5.create_dataset(
        "synch_coeff", data=synch_coeff, compression="gzip", compression_opts=9
    )

     # get other properties from h5py file
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]

    low_cor, up_cor = bbox[0, :, 0], bbox[-1, :, 1]

    # # compute synchrotron luminosity
    # now we need to add astropy units since we would once convert to pc
    # as cm^3 with 1e21 overloads
    cell_size_data = (
        np.tile(bsize[:, 0] / 8.0, reps=(8, 8, 8)).reshape((bsize.shape[0], 8, 8, 8))
        * u.cm
    ).to_value(u.pc)

    synch_lum_per_freq = []

    for synch_freq in synch_freqs:
        synch_freq_ratio = (synch_freq / nu_crit_data).to_value(u.dimensionless_unscaled)
        synch_kernel_data = synch_kernel(synch_freq_ratio)
        del synch_freq_ratio

        synch_output_h5.create_dataset(
            f"synch_kernel_{synch_freq.to(u.GHz).value:.2f}GHz",
            data=synch_kernel_data,
            compression="gzip",
            compression_opts=9,
        )

        ncre_synch = np.trapz(
            4 * np.pi * p_arr[:, None, None, None] ** 2 * f_e_data * synch_kernel_data,
            x=p_arr[:, None, None, None],
            axis=0,
        ) / ncr_e
        ncre_synch_data = (ncre_synch / ecr_p * encr_data).to_value(u.cm**-3)
        synch_output_h5.create_dataset(
            f"ncre_synch_{synch_freq.to(u.GHz).value:.2f}GHz",
            data=ncre_synch_data,
            compression="gzip",
            compression_opts=9,
        )

        synch_intensity_data = (
            synch_coeff
            * ncre_synch_data
            * u.erg / u.s / u.Hz / u.cm**3
        ).to_value(u.erg / u.s / u.Hz / u.cm**3)
        synch_output_h5.create_dataset(
            f"synch_intensity_{synch_freq.to(u.GHz).value:.2f}GHz",
            data=synch_intensity_data,
            compression="gzip",
            compression_opts=9,
        )

        # luminosity
        synch_lum = np.sum(
            (synch_intensity_data * (u.erg / u.s / u.Hz / u.cm**3)).to_value(u.erg / u.s / u.pc**3) * cell_size_data**3,
            axis=(0, 1, 2, 3),
        )
        print(f"Synchrotron luminosity at {synch_freq.to(u.GHz).value:.2f} GHz: {synch_lum:.2e} erg/s/Hz")
        synch_lum_per_freq.append(synch_lum)

        del synch_intensity_data
        del ncre_synch_data
        del synch_kernel_data

    del f_e_data

    synch_output_h5.close()

   
    # # specific synchrotron intensity is in -> E dN/(dV dt), in units of erg / s / Hz / cm^3
    # # NB: astropy doesnt properly take care of cgs units with magnetic fields, so we convert them to values instead and add units later
    # synch_coeff = (
    #     np.sqrt(3)
    #     * Blos_data  # LoS B-field in Gauss
    #     * const.e.gauss.value**3
    #     / (const.m_e.cgs.value * const.c.cgs.value**2)
    # )

    # synch_lum_per_freq = []
    # for ifreq, synch_freq in tqdm(
    #     enumerate(synch_freq_grid), total=len(synch_freq_grid)
    # ):
    #     # if ifreq % 2 != 0:
    #     #     continue

    #     # synchrotron intensity in  (u.erg * u.s**-1 * u.Hz**-1 * u.cm**-3)
    #     synch_intens_data = (
    #         synch_coeff
    #         * pf["encr"][()][blist]  # specific CR energy
    #         * pf["dens"][()][blist]  # density
    #     ) / (
    #         (
    #             (
    #                 10.0
    #                 ** f_log10_ensynchratio(
    #                     (np.log10(dens_data), np.log10(Blos_data)),
    #                 )[..., ifreq]
    #             )
    #             * u.GeV
    #         ).to_value(u.erg)
    #     )  # need to convert this to erg

    #     for t in times_to_store:
    #         if np.isclose(t, time.to_value(u.kyr), atol=0.1):  # in 30 kyr intervals
    #             print(t, time.to_value(u.kyr))
    #             synch_idx = np.digitize(1.5 * u.GHz, synch_freq_grid, right=True)
    #             if ifreq != synch_idx:
    #                 continue
    #             synch_intens_sl = flash_amr_tools.get_slice(
    #                 synch_intens_data * synch_freq.to_value(u.Hz),
    #                 pos=0.0,
    #                 axis=2,
    #                 ref_lvl=ref_lvl,
    #                 bbox=bbox,
    #                 bsize=bsize,
    #                 brefs=brefs,
    #                 bns=bns,
    #             )
    #             # write data to h5py file
    #             with h5py.File(
    #                 f"{data_dir}/emission_data_synch_timecorrected_ae{args.alpha_inj_e}.h5",
    #                 "a",
    #             ) as pfo:
    #                 sim_gr = (
    #                     pfo.create_group(sim_name)
    #                     if sim_name not in pfo.keys()
    #                     else pfo[sim_name]
    #                 )

    #                 synch_sl_dr = (
    #                     sim_gr["slices"]
    #                     if "slices" in sim_gr.keys()
    #                     else sim_gr.create_group("slices")
    #                 )

    #                 if f"t{t}kyr" in synch_sl_dr.keys():
    #                     del synch_sl_dr[f"t{t}kyr"]
    #                 if f"t{t}_bbox" in synch_sl_dr.keys():
    #                     del synch_sl_dr[f"t{t}_bbox"]

    #                 synch_sl_dr.create_dataset(f"t{t}kyr", data=synch_intens_sl)
    #                 synch_sl_dr.create_dataset(f"t{t}_bbox", data=bbox)

    #             del synch_intens_sl

    #     synch_lum_per_freq.append(
    #         np.sum(
    #             (synch_intens_data * (u.erg / u.s / u.Hz / u.cm**3)).to_value(
    #                 u.erg / u.s / u.Hz / u.pc**3
    #             )
    #             * cell_size_data**3,
    #             axis=(0, 1, 2, 3),
    #         )
    #     )

    #     del synch_intens_data

    # del cell_size_data
    # del synch_coeff
    # del Blos_data


    print(np.array(synch_lum_per_freq) * synch_freqs.to_value(u.Hz))

    times.append(time.to_value(u.s))
    synch_lum_arr.append(synch_lum_per_freq)

    pf.close()

    print(
        "Memory Usage: ",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        " MiB",
    )

times = np.array(times) * u.s
synch_lum_arr = np.array(synch_lum_arr) * (u.erg / u.s / u.Hz)

print(synch_lum_arr.min(), synch_lum_arr.max())

# sort arrays based on increasing order in time
synch_lum_arr = synch_lum_arr[np.argsort(times), :]
times = np.sort(times)

# write data to h5py file
with h5py.File(
    f"{data_dir}/emission_data_synch.h5", "a"
) as pf:
    sim_gr = pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]

    if "times" in pf[sim_name].keys():
        del pf[sim_name]["times"]
    sim_gr.create_dataset("times", data=times.to(u.kyr))

    if "synch_lum" in pf[sim_name].keys():
        del pf[sim_name]["synch_lum"]
    sim_gr.create_dataset("synch_lum", data=synch_lum_arr.to(u.erg / u.s / u.Hz))

    if "synch_freq" in pf[sim_name].keys():
        del pf[sim_name]["synch_freq"]
    sim_gr.create_dataset("synch_freq", data=synch_freqs.to_value(u.GHz))
