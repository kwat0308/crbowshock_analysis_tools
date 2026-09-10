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

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/updated_outputs"
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

parser.add_argument(
    "--chunk_size",
    action="store",
    type=int,
    help="chunk size",
    default=50
)

args = parser.parse_args()

# here we define the normalised momentum grid for both protons and electrons,

# pp_arr = np.logspace(0.8, 4, 200) * u.GeV / c  # energy grid for protons
# pp_arr = (pp_arr / (mp * c)).to_value(
#     u.dimensionless_unscaled
# )  # normalised momentum wrt protons
pp_arr = np.logspace(0., 5, 200)
print(
    f"Minimum & maximum of normalised momentum (protons): {(pp_arr.min() * mp * c**2).to(u.GeV):.5e}, {(pp_arr.max() * mp * c**2).to(u.GeV):.5e}"
)

# pe_arr = np.logspace(-5, 4, 200) * u.GeV / c # energy grid for electrons
# pe_arr = (pe_arr / (me * c)).to_value(
#     u.dimensionless_unscaled
# )  # normalised momentum wrt protons
pe_arr = np.logspace(2, 7, 200)
print(
    f"Minimum & maximum of normalised momentum (electrons): {(pe_arr.min() * me * c**2).to(u.GeV):.5e}, {(pe_arr.max() * me * c**2).to(u.GeV):.5e}"
)

Kep, pp0, pe0 = (
    ep_ratio()
)  # get the normalisation for the electron spectrum based on the proton spectrum at 10 GeV, and the corresponding momenta

synch_freqs = np.array([
    0.5, 1.5, 3.0, 5.0, 10.0, 15.0
]) * u.GHz  # in Hz

sim_name = args.sim_name
chunk_size = args.chunk_size

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

    if time > 505 * u.kyr:
        print(f"Time {time.to(u.kyr)} > 500 kyr, ending computation")
        pf.close()
        break

    print(f"Processing {sim_file}, time = {time.to(u.kyr)}")

    # loop over each file and get the slice
    blist, brefs, bns = flash_amr_tools.get_true_blocks(
        f"{flash_output_dir}/{sim_dir}/{sim_file}",
    )

    # LoS magnetic field (in Gauss)  ==  z-axis -> transverse is Bx and By contributions
    # Blos_data = np.abs(pf["magx"][()][blist]) + np.abs(pf["magy"][()][blist])
    Blos_data = (
        (2 / 3)
        * np.sqrt(
            pf["magx"][()][blist] ** 2
            + pf["magy"][()][blist] ** 2
            + pf["magz"][()][blist] ** 2
        )
        * u.G
    )
    Blos_data[Blos_data < 1e-30 * u.G] = 1e-30 * u.G  # set a floor to avoid issues
    dens_data = pf["dens"][()][blist] * u.g / u.cm**3
    dens_data[dens_data < 1e-40 * u.g / u.cm**3] = (
        1e-40 * u.g / u.cm**3
    )  # set a floor to avoid issues
    vel_data = np.sqrt(
        pf["velx"][()][blist] ** 2
        + pf["vely"][()][blist] ** 2
        + pf["velz"][()][blist] ** 2
    ) * (u.cm / u.s)
    encr_data = (
        (pf["encr"][()][blist] * pf["dens"][()][blist]) * u.erg / u.cm**3
    )  #  CR energy density in erg/cm^3

    # computing cell size
    bbox = pf["bounding box"][()][blist]
    node_type = pf["node type"][()][blist]
    low_corner = bbox[:, :, 0]
    cell_dx = pf["block size"][()][blist][:, 0] / 8
    cell_dy = pf["block size"][()][blist][:, 1] / 8
    cell_dz = pf["block size"][()][blist][:, 2] / 8
    cell_d = np.array((cell_dx, cell_dy, cell_dz))

    # meshgrid of cell sizes
    k, j, i = np.meshgrid(range(8), range(8), range(8), indexing="ij")

    # mask to get all leaf nodes (those with max refinement in each block)
    leafmask = node_type == 1

    # x coordinate at lead node
    xcoord = low_corner[leafmask, 0, None, None, None] + cell_dx[
        leafmask, None, None, None
    ] * (
        0.5 + i[None,]
    )
    ycoord = low_corner[leafmask, 1, None, None, None] + cell_dy[
        leafmask, None, None, None
    ] * (
        0.5 + j[None,]
    )
    zcoord = low_corner[leafmask, 2, None, None, None] + cell_dz[
        leafmask, None, None, None
    ] * (
        0.5 + k[None,]
    )
    rcoord = np.sqrt(xcoord**2 + ycoord**2 + zcoord**2) * u.cm
    del (
        xcoord,
        ycoord,
        zcoord,
        leafmask,
        i,
        j,
        k,
        cell_dx,
        cell_dy,
        cell_dz,
        low_corner,
        bbox,
    )

    synch_output_file = f"{output_dir}/emission_data_t{time.to(u.kyr).value:.0f}.h5"
    synch_output_h5 = h5py.File(synch_output_file, "a")

    # # now for synchrotron emission related stuff
    # nu_crit_data = crit_freq(Blos=Blos_data[None, :, :, :, :], pele=pe_arr[:, None, None, None, None])
    # print(f"Critial Frequency in Hz: {nu_crit_data.min():.2e}, {nu_crit_data.max():.2e}")
    # synch_output_h5.create_dataset(
    #     "nu_crit", data=nu_crit_data.value, compression="lzf", dtype='float32', shape=nu_crit_data.shape, chunks=(200, 100, 8, 8, 8)
    # )
    synch_coeff = (
        np.sqrt(3)
        * Blos_data.value  # LoS B-field in Gauss
        * const.e.gauss.value**3
        / (const.m_e.cgs.value * const.c.cgs.value**2)
    )
    print(
        f"Sycnrhtron Emission Strength: {synch_coeff.min():.2e}, {synch_coeff.max():.2e}"
    )
    # if "synch_coeff" in synch_output_h5:
    #     del synch_output_h5['synch_coeff']
    # synch_output_h5.create_dataset(
    #     "synch_coeff", data=synch_coeff, compression="lzf", dtype='float32', shape=synch_coeff.shape, chunks=(100, 8, 8, 8)
    # )

    # recalculate this value
    ecr_p_val = np.zeros(dens_data.shape) * (u.s**(11/5) / (u.cm**(6/5) * u.g**(11/5)))
    ncr_p_val = np.zeros(dens_data.shape) * (u.s**(21/5) / (u.cm**(16/5) * u.g**(16/5)))

    for pp_start in range(0, len(pp_arr), chunk_size):
        print(f"Calculating ncr and ecr for protons, momentum bin {pp_start} to {pp_start+chunk_size}")
        pp_end = min(pp_start + chunk_size, len(pp_arr))
        pp_chunk = pp_arr[pp_start:pp_end]

        f_p_0_chunk = f_p_0(
            args.alpha_inj_p,
            pp_chunk[:, None, None, None, None],
            dens_data[None, :, :, :, :],
            vel_data[None, :, :, :, :],
            pmax=bowshock_config[sim_name]["pmax_prot"],
            vstar=bowshock_config[sim_name]["vstar"] * (u.km/u.s)
        )

        ncr_p_chunk = np.trapz(
            4 * np.pi * pp_chunk[:, None, None, None, None] ** 2 * f_p_0_chunk,
            x=pp_chunk[:, None, None, None, None],
            axis=0,
        )

        ecr_p_chunk = np.trapz(
            4 * np.pi * pp_chunk[:, None, None, None, None] ** 2 * ekin(pp_chunk[:, None, None, None, None]) * f_p_0_chunk,
            x=pp_chunk[:, None, None, None, None],
            axis=0,
        ) * (mpc2)

        ecr_p_val += ecr_p_chunk.sum(axis=0)  # this is just the normalisation factor for the CR energy density, which we will use to renormalise the electron spectrum based on the ep ratio at 10 GeV

        ncr_p_val += ncr_p_chunk.sum(axis=0)
        del f_p_0_chunk, ecr_p_chunk, ncr_p_chunk

    # print(ecr_p_val.min(), ecr_p_val.max())
    ecr_p = (ecr_p_val / ncr_p_val).to(u.GeV)
    print(ecr_p.min(), ecr_p.max())
    del ncr_p_val, ecr_p_val

    ncr_e_val = np.zeros(dens_data.shape) * (u.s**(21/5) / (u.cm**(16/5) * u.g**(16/5)))
    for pe_start in range(0, len(pe_arr), chunk_size):
        print(
            f"Calculating ncr and ecr for electrons, momentum bin {pe_start} to {pe_start+chunk_size}"
        )
        pe_end = min(pe_start + chunk_size, len(pe_arr))
        pe_chunk = pe_arr[pe_start:pe_end]

        f_e_0_chunk = f_e_0(
            args.alpha_inj_e,
            pe_chunk[:, None, None, None, None],
            dens_data[None, :, :, :, :],
            Blos_data[None, :, :, :, :],
            Tstar=1e4 * u.K,  # assumed based on stellar track from Ekstrom+2012
            vel=vel_data[None, :, :, :, :],
            pmax=bowshock_config[sim_name]["pmax_electr"],
            vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
            r_pos=rcoord[None, ...],
        )

        # for electrons, need to renormalise the spectrum based on the ep ratio at 10 GeV
        f_e_chunk = (
            Kep
            * f_p_0(
                args.alpha_inj_p,
                pp0,
                dens_data,
                vel_data,
                pmax=bowshock_config[sim_name]["pmax_prot"],
                vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
            )
            / f_e_0(
                args.alpha_inj_e,
                pe0,
                dens_data,
                Blos_data,
                Tstar=1e4 * u.K,
                vel=vel_data,
                pmax=bowshock_config[sim_name]["pmax_electr"],
                vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
                r_pos=rcoord[None, ...],
            )
            * f_e_0_chunk
        )
        del f_e_0_chunk

        ncr_e_chunk = np.trapz( 4 * np.pi * pe_chunk[:, None, None, None, None] ** 2 * f_e_chunk,
            x=pe_chunk[:, None, None, None, None],
            axis=0,
        )

        ncr_e_val += ncr_e_chunk.sum(axis=0)
        del f_e_chunk, ncr_e_chunk

    print(f"Number density of CR electrons in a.u.: {ncr_e_val.min().value:.2e}, {ncr_e_val.max().value:.2e}")
    # break

    synch_lum_per_freq = []

    for synch_freq in synch_freqs:
        
        ncre_synch_data = np.zeros(dens_data.shape)
        for pe_start in range(0, len(pe_arr), chunk_size):
            print(
                f"Calculating ncr and ecr for synch. electrons, momentum bin {pe_start} to {pe_start+chunk_size}"
            )
            pe_end = min(pe_start + chunk_size, len(pe_arr))
            pe_chunk = pe_arr[pe_start:pe_end]

            f_e_0_chunk = f_e_0(
                args.alpha_inj_e,
                pe_chunk[:, None, None, None, None],
                dens_data[None, :, :, :, :],
                Blos_data[None, :, :, :, :],
                Tstar=1e4 * u.K,  # assumed based on stellar track from Ekstrom+2012
                vel=vel_data[None, :, :, :, :],
                pmax=bowshock_config[sim_name]["pmax_electr"],
                vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
                r_pos=rcoord[None, ...],
            )

            # for electrons, need to renormalise the spectrum based on the ep ratio at 10 GeV
            f_e_chunk = (
                Kep
                * f_p_0(
                    args.alpha_inj_p,
                    pp0,
                    dens_data,
                    vel_data,
                    pmax=bowshock_config[sim_name]["pmax_prot"],
                    vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
                )
                / f_e_0(
                    args.alpha_inj_e,
                    pe0,
                    dens_data,
                    Blos_data,
                    Tstar=1e4 * u.K,
                    vel=vel_data,
                    pmax=bowshock_config[sim_name]["pmax_electr"],
                    vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
                    r_pos=rcoord[None, ...],
                )
                * f_e_0_chunk
            )
            del f_e_0_chunk
            ### CALCULATE THE NCREEEEE
            # THEN NORMALISE

            nu_crit_data = crit_freq(
                Blos=Blos_data[None, :, :, :, :],
                pele=pe_chunk[:, None, None, None, None],
            )
            synch_freq_ratio = (synch_freq / nu_crit_data).to_value(
                u.dimensionless_unscaled
            )
            synch_kernel_data = synch_kernel(synch_freq_ratio)
            del synch_freq_ratio
            # print(
            #     f"Synchrotron Kernal at {synch_freq}: {synch_kernel_data.min():.2e}, {synch_kernel_data.max():.2e}"
            # )
            # within_integral = 4 * np.pi * pe_chunk[:, None, None, None, None] ** 2 * f_e_chunk * synch_kernel_data
            # print(f"Within integral for ncre_synch at {synch_freq}: {within_integral.min():.2e}, {within_integral.max():.2e}")
            
            # effectively a weighted sum of electrons, ratioed by the 
            # cosmic ray proton energy
            # normalize by the total number fraction of electrons
            ncre_synch = np.trapz(
                4
                * np.pi
                * pe_chunk[:, None, None, None, None] ** 2
                * f_e_chunk
                * synch_kernel_data,
                x=pe_chunk[:, None, None, None, None],
                axis=0,
            ) / ncr_e_val

            ncre_synch_data += ncre_synch.sum(axis=0)
            del f_e_chunk, ncre_synch

        ncre_synch_data = (ncre_synch_data / ecr_p * encr_data).to_value(u.cm**-3)

        print(
            f"CR electrons undergoing synchrotron emission at {synch_freq}: {ncre_synch_data.min():.2e}, {ncre_synch_data.max():.2e}"
        )
        ncre_synch_label = f"ncre_synch_{synch_freq.to(u.GHz).value:.2f}GHz"
        if ncre_synch_label in synch_output_h5:
            del synch_output_h5[ncre_synch_label]
        synch_output_h5.create_dataset(
            ncre_synch_label,
            data=ncre_synch_data,
            compression="lzf",
            dtype="float32",
            shape=ncre_synch_data.shape,
            chunks=(100, 8, 8, 8),
        )

        synch_intensity_data = (
            synch_coeff * ncre_synch_data * u.erg / u.s / u.Hz / u.cm**3
        ).to_value(u.erg / u.s / u.Hz / u.cm**3)
        print(
            f"Synchrotron intensity at {synch_freq} (erg/s/Hz/cm^3): {synch_intensity_data.min():.2e}, {synch_intensity_data.max():.2e}"
        )
        synch_intens_label = f"synch_intensity_{synch_freq.to(u.GHz).value:.2f}GHz"
        if synch_intens_label in synch_output_h5:
            del synch_output_h5[synch_intens_label]
        synch_output_h5.create_dataset(
            synch_intens_label,
            data=synch_intensity_data,
            compression="lzf",
            dtype="float32",
            shape=synch_intensity_data.shape,
            chunks=(100, 8, 8, 8),
        )
        del ncre_synch_data

        # get other properties from h5py file
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]

        low_cor, up_cor = bbox[0, :, 0], bbox[-1, :, 1]

        # compute synchrotron luminosity
        cell_size_data = (
            np.tile(bsize[:, 0] / 8.0, reps=(8, 8, 8)).reshape(
                (bsize.shape[0], 8, 8, 8)
            )
            * u.cm
        ).to_value(u.pc)

        # luminosity
        synch_lum = np.sum(
            (synch_intensity_data * (u.erg / u.s / u.Hz / u.cm**3)).to_value(
                u.erg / u.s / u.Hz / u.pc**3
            )
            * cell_size_data**3,
            axis=(0, 1, 2, 3),
        )
        print(
            f"Synchrotron luminosity at {synch_freq.to(u.GHz).value:.2f} GHz: {synch_lum:.2e} erg/s/Hz"
        )
        synch_lum_per_freq.append(synch_lum)

        del synch_intensity_data
        del synch_kernel_data

    synch_output_h5.close()
    del ecr_p
    del ncr_e_val

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
with h5py.File(f"{data_dir}/emission_data_synch_term.h5", "a") as pf:
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
