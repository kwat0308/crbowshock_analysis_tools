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

pp_arr = np.logspace(0, 4, 200)
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
    Blos_data = (2 / 3) * np.sqrt(
        pf["magx"][()][blist] ** 2
        + pf["magy"][()][blist] ** 2
        + pf["magz"][()][blist] ** 2
    ) * u.G
    Blos_data[Blos_data < 1e-30 * u.G] = 1e-30 * u.G  # set a floor to avoid issues
    dens_data = pf["dens"][()][blist] * u.g/u.cm**3
    dens_data[dens_data < 1e-40* u.g/u.cm**3] = 1e-40* u.g/u.cm**3  # set a floor to avoid issues
    vel_data = np.sqrt(
        pf["velx"][()][blist] ** 2
        + pf["vely"][()][blist] ** 2
        + pf["velz"][()][blist] ** 2
    ) * (u.cm/u.s)
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

    # number density also
    # this is calcualted by the ratio of n / ecr * ecr_data, where n is the number density of CRs calculated by integrating over the momentum grid, and ecr_data is the CR energy density from the simulation data
    ncr_p_data = np.zeros(dens_data.shape)
    ecr_p_val = np.zeros(dens_data.shape) * (u.s**(11/5) / (u.cm**(6/5) * u.g**(11/5)))

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

        ncr_p_chunk = np.trapz( 4 * np.pi * pp_chunk[:, None, None, None, None] ** 2 * f_p_0_chunk,
            x=pp_chunk[:, None, None, None, None],
            axis=0,
        )
        ecr_p_chunk = np.trapz(
            4 * np.pi * pp_chunk[:, None, None, None, None] ** 2 * ekin(pp_chunk[:, None, None, None, None]) * f_p_0_chunk,
            x=pp_chunk[:, None, None, None, None],
            axis=0,
        ) * (mpc2)

        ncr_p_data += ncr_p_chunk.sum(axis=0)
        ecr_p_val += ecr_p_chunk.sum(axis=0)  # this is just the normalisation factor for the CR energy density, which we will use to renormalise the electron spectrum based on the ep ratio at 10 GeV
        del f_p_0_chunk, ncr_p_chunk, ecr_p_chunk

    ncr_p_data = (ncr_p_data / ecr_p_val * encr_data).to_value(u.cm**-3)

    print(f"Number density of CR protons in cm^-3: {ncr_p_data.min():.2e}, {ncr_p_data.max():.2e}")
    print(f"Energy density of CR protons in a.u.: {ecr_p_val.min():.2e}, {ecr_p_val.max():.2e}")

    if "ncr_p" in synch_output_h5:
        del synch_output_h5['ncr_p']
    synch_output_h5.create_dataset(
        "ncr_p", data=ncr_p_data, compression="lzf", dtype='float32', shape=ncr_p_data.shape, chunks=(100, 8, 8, 8)
    )
    del ncr_p_data

    print("Computed CR proton Properties")

    # number density also
    # this is calcualted by the ratio of n / ecr * ecr_data, where n is the number density of CRs calculated by integrating over the momentum grid, and ecr_data is the CR energy density from the simulation data
    ncr_e_data = np.zeros(dens_data.shape)

    for pe_start in range(0, len(pe_arr), chunk_size):
        print(f"Calculating ncr and ecr for electrons, momentum bin {pe_start} to {pe_start+chunk_size}")
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
            vstar=bowshock_config[sim_name]["vstar"] * (u.km/u.s),
            r_pos=rcoord[None,...]
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
                vstar=bowshock_config[sim_name]["vstar"] * (u.km/u.s)
            )
            / f_e_0(
                args.alpha_inj_e,
                pe0,
                dens_data,
                Blos_data,
                Tstar=1e4 * u.K,
                vel=vel_data,
                pmax=bowshock_config[sim_name]["pmax_electr"],
                vstar=bowshock_config[sim_name]["vstar"] * (u.km/u.s),
                r_pos=rcoord[None,...]
            )
            * f_e_0_chunk
        )
        del f_e_0_chunk

        ncr_e_chunk = np.trapz( 4 * np.pi * pe_chunk[:, None, None, None, None] ** 2 * f_e_chunk,
            x=pe_chunk[:, None, None, None, None],
            axis=0,
        )

        ncr_e_data += ncr_e_chunk.sum(axis=0)

        del f_e_chunk, ncr_e_chunk

    ncr_e_data = (ncr_e_data / ecr_p_val * encr_data).to_value(u.cm**-3)

    print(f"Number density of CR electrons in cm^-3: {ncr_e_data.min():.2e}, {ncr_e_data.max():.2e}")
    # print(f"Energy density of CR electrons in erg cm^-3: {encr_e_data.min():.2e}, {encr_e_data.max():.2e}")
    if "ncr_e" in synch_output_h5:
        del synch_output_h5['ncr_e']
    synch_output_h5.create_dataset(
        "ncr_e", data=ncr_e_data, compression="lzf", dtype='float32', shape=ncr_e_data.shape, chunks=(100, 8, 8, 8)
    )
    del ncr_e_data

    synch_output_h5.close()

    pf.close()

    print(
        "Memory Usage: ",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        " MiB",
    )

