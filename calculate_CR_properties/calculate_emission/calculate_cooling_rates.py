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

pp_arr = np.logspace(0.8, 4, 200) * u.GeV / c  # energy grid for protons
pp_arr = (pp_arr / (mp * c)).to_value(
    u.dimensionless_unscaled
)  # normalised momentum wrt protons
print(
    "Minimum & maximum of normalised momentum (protons): ", pp_arr.min(), pp_arr.max()
)

pe_arr = np.logspace(-2, 4, 200) * u.GeV / c  # energy grid for electrons
pe_arr = (pe_arr / (me * c)).to_value(
    u.dimensionless_unscaled
)  # normalised momentum wrt protons
print(
    "Minimum & maximum of normalised momentum (electrons): ", pe_arr.min(), pe_arr.max()
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
    # if os.path.exists(synch_output_file):
    #     continue
    synch_output_h5 = h5py.File(synch_output_file, "a")

    b_hadr_data = np.zeros(dens_data.shape)
    b_coul_data = np.zeros(dens_data.shape)
    b_adiab_data = np.zeros(dens_data.shape)

    for pp_start in range(0, len(pp_arr), chunk_size):
        print(f"Calculating cooling for protons, momentum bin {pp_start} to {pp_start+chunk_size}")
        pp_end = min(pp_start + chunk_size, len(pp_arr))
        pp_chunk = pp_arr[pp_start:pp_end]

        b_hadr_chunk = b_hadr(
            pp_chunk[:,None,None,None,None], 
            dens_data[None, :, :, :, :]
        ).astype(np.float32)
        b_hadr_data += b_hadr_chunk.to_value(1 / u.s).sum(axis=0)
        del b_hadr_chunk

        b_coul_chunk = b_coul_p(
            pp_chunk[:, None, None, None, None], dens_data[None, :, :, :, :]
        ).astype(np.float32)
        b_coul_data += b_coul_chunk.to_value(1 / u.s).sum(axis=0)
        del b_coul_chunk

        b_adiab_chunk = b_adiab(
            pp_chunk[:, None, None, None, None],
            vel_data[None, :, :, :, :],
            vstar=bowshock_config[sim_name]["vstar"] * u.km / u.s,
        ).astype(np.float32)
        b_adiab_data += b_adiab_chunk.to_value(1 / u.s).sum(axis=0)
        del b_adiab_chunk

    b_hadr_data /= len(pp_arr)
    print(f"Hadronic Cooling: {b_hadr_data.min():.2e}, {b_hadr_data.max():.2e}")
    if "b_hadr" in synch_output_h5:
        del synch_output_h5['b_hadr']
    synch_output_h5.create_dataset(
        "b_hadr",
        data=b_hadr_data,
        compression="lzf",
        dtype="float32",
        shape=b_hadr_data.shape,
        chunks=(100, 8, 8, 8),
    )
    del b_hadr_data

    b_coul_data /= len(pp_arr)

    print(f"Coulomb Cooling: {b_coul_data.min():.2e}, {b_coul_data.max():.2e}")
    if "b_coul" in synch_output_h5:
        del synch_output_h5['b_coul']
    synch_output_h5.create_dataset(
        "b_coul",
        data=b_coul_data,
        compression="lzf",
        dtype="float32",
        shape=b_coul_data.shape,
        chunks=(100, 8, 8, 8),
    )
    del b_coul_data

    b_adiab_data /= len(pp_arr)
    print(f"Adiabatic Cooling: {b_adiab_data.min():.2e}, {b_adiab_data.max():.2e}")
    if "b_adiab" in synch_output_h5:
        del synch_output_h5['b_adiab']
    synch_output_h5.create_dataset(
        "b_adiab",
        data=b_adiab_data,
        compression="lzf",
        dtype="float32",
        shape=b_adiab_data.shape,
        chunks=(100, 8, 8, 8),
    )
    del b_adiab_data

    b_synch_data = np.zeros(dens_data.shape)
    # b_ic_photon_data = np.zeros(dens_data.shape)
    b_ic_star_data = np.zeros(dens_data.shape)
    b_coul_e_data = np.zeros(dens_data.shape)

    for pe_start in range(0, len(pe_arr), chunk_size):
        print(f"Calculating cooling for electrons, momentum bin {pe_start} to {pe_start+chunk_size}")
        pe_end = min(pe_start + chunk_size, len(pe_arr))
        pe_chunk = pe_arr[pe_start:pe_end]

        b_synch_chunk = b_synch(
            pe_chunk[:, None, None, None, None], Blos_data[None, :, :, :, :]
        ).astype(np.float32)

        b_synch_data += b_synch_chunk.to_value(1 / u.s).sum(axis=0)
        del b_synch_chunk

        # b_ic_photon_chunk = b_ic_photon(
        #     pe_chunk[:, None, None, None, None]
        # ).astype(np.float32)

        # b_ic_photon_data += b_ic_photon_chunk.to_value(1 / u.s).sum(axis=0)
        # del b_ic_photon_chunk

        b_ic_star_chunk = b_ic_star(
            pe_chunk[:, None, None, None, None], Tstar=1e4 * u.K, r_pos=rcoord
        ).astype(np.float32)

        b_ic_star_data += b_ic_star_chunk.to_value(1 / u.s).sum(axis=0)
        del b_ic_star_chunk

        b_coul_e_chunk = b_coul_e(
            pe_chunk[:, None, None, None, None], dens_data[None, :, :, :, :]
        ).astype(np.float32)

        b_coul_e_data += b_coul_e_chunk.to_value(1 / u.s).sum(axis=0)
        del b_coul_e_chunk

    b_synch_data /= len(pe_arr)
    print(f"Synchrotron Cooling: {b_synch_data.min():.2e}, {b_synch_data.max():.2e}")
    if "b_synch" in synch_output_h5:
        del synch_output_h5['b_synch']
    synch_output_h5.create_dataset(
        "b_synch",
        data=b_synch_data,
        compression="lzf",
        dtype="float32",
        shape=b_synch_data.shape,
        chunks=(100, 8, 8, 8),
    )
    del b_synch_data
    # print(
    #     f"IC Cooling from Background photons: {b_ic_photon_data.min():.2e}, {b_ic_photon_data.max():.2e}"
    # )
    # if "b_ic_photon" in synch_output_h5:
    #     del synch_output_h5['b_ic_photon']
    # synch_output_h5.create_dataset(
    #     "b_ic_photon",
    #     data=b_ic_photon_data,
    #     compression="lzf",
    #     dtype="float32",
    #     shape=b_ic_photon_data.shape,
    # )
    # del b_ic_photon_data
    b_ic_star_data /= len(pe_arr)
    print(
        f"IC Cooling from Stellar photons: {b_ic_star_data.min():.2e}, {b_ic_star_data.max():.2e}"
    )
    if "b_ic_star" in synch_output_h5:
        del synch_output_h5['b_ic_star']
    synch_output_h5.create_dataset(
        "b_ic_star",
        data=b_ic_star_data,
        compression="lzf",
        dtype="float32",
        shape=b_ic_star_data.shape,
        chunks=(100, 8, 8, 8),
    )
    del b_ic_star_data
    print(
        f"Coulomb Cooling for electrons: {b_coul_e_data.min():.2e}, {b_coul_e_data.max():.2e}"
    )
    if "b_coul_e" in synch_output_h5:
        del synch_output_h5['b_coul_e']
    synch_output_h5.create_dataset(
        "b_coul_e",
        data=b_coul_e_data,
        compression="lzf",
        dtype="float32",
        shape=b_coul_e_data.shape,
    )
    del b_coul_e_data
    print("Calculated energy loss terms")

    pf.close()
    synch_output_h5.close()

    print(
        "Memory Usage: ",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        " MiB",
    )
