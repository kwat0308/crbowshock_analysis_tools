import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import h5py
import os
from scipy.interpolate import CubicSpline

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import flash_amr_tools

import json

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs_new"
fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# load the bowshock config
bowshock_config = json.load(open("../../config/bowshock_config.json", "r"))["config"]

import argparse

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
    "--pos", action="store", type=float, help="Position of slice", default=0.0
)
parser.add_argument("--axis", action="store", type=int, help="Axis of slice", default=2)
args = parser.parse_args()

sim_name = args.sim_name
sim_dir = bowshock_config[args.sim_name]["sim_dir"]

with h5py.File(f"{data_dir}/steady_state_spectrum.h5", "r") as f:
    alpha_inj_grid = f["alpha_inj_grid"][()]
    log10_enratio_grid = f["proton/log10_enpratio_grid"][()]

f_log10_enratio_grid = CubicSpline(x=alpha_inj_grid, y=log10_enratio_grid)


if not os.path.exists(f"{fig_dir}/{sim_name}"):
    os.mkdir(f"{fig_dir}/{sim_name}")

# get list of all files
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
    if file.find("plt") < 0:
        continue
    sim_files.append(file)


# sort
sim_files = sorted(sim_files)[::-1]

# truncate to every 10 plot files if we have over 200 plot files
if len(sim_files) > 200:
    sim_files = sim_files[::10]

# store total number of CRs per time
ncr_tot_arr = np.zeros(len(sim_files))
times = np.zeros(len(sim_files)) * u.s

# loop over each file and get the slice
for isim, sim_file in enumerate(sim_files):

    blist, brefs, bns = flash_amr_tools.get_true_blocks(
        f"{flash_output_dir}/{sim_dir}/{sim_file}"
    )

    with h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r") as pf:
        shma_data = pf["shma"][()][blist]
        dens_data = pf["dens"][()][blist] * (u.g / u.cm**3)
        encr_data = pf["encr"][()][blist] * (u.erg / u.g)
        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]
        gam = pf["geff"][()][blist]

        times[isim] = pf["real scalars"][()][0][1] * u.s  # time in seconds (this is fine)

    low_cor, up_cor = bbox[0, :, 0], bbox[-1, :, 1]

    # calculate injection index
    # gam = 5/3

    xs_data = (gam + 1) * shma_data**2 / (2 + (gam - 1) * shma_data**2)
    alpha_inj_data = 3 * xs_data / (xs_data - 1)
    # force minimal and maximal values
    alpha_inj_data[alpha_inj_data < 4] = 4  # soft spectral index for background CRs
    alpha_inj_data[alpha_inj_data >= 10] = 10  # similarly

    # compute the enratio to get CR number density
    enratio_data = (10.0 ** f_log10_enratio_grid(alpha_inj_data))* u.GeV

    ncr_data = (encr_data * dens_data / enratio_data).to(u.cm**-3)

    # # compute total number of CR electrons
    # now we need to add astropy units since we would once convert to pc
    # as cm^3 with 1e21 overloads
    cell_size_data = (
        np.tile(bsize[:, 0] / 8.0, reps=(8, 8, 8)).reshape((bsize.shape[0], 8, 8, 8))
        * u.cm
    )
    ncr_tot_arr[isim] = np.sum( ncr_data.to(u.pc**-3) * cell_size_data.to(u.pc) ** 3)

    if isim % 5 != 0:
        ncre_sl = flash_amr_tools.get_slice(
            ncr_data,
            pos=args.pos,
            axis=args.axis,
            ref_lvl=ref_lvl,
            bbox=bbox,
            bsize=bsize,
            brefs=brefs,
            bns=bns,
        )
        print(ncre_sl.min(), ncre_sl.max())

        # write data to h5py file
        with h5py.File(f"{data_dir}/emission_data_ncr.h5", "a") as pf:
            sim_gr = (
                pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]
            )

            if "electron" in sim_gr.keys():
                del sim_gr["electron"]

            sim_ele_gr = sim_gr.create_group("electron")

            if "slices" in sim_ele_gr.keys():
                del sim_ele_gr["slices"]
            ncr_sl_dr = sim_ele_gr.create_group("slices")
            ncr_sl_dr.create_dataset(f"t{isim}0kyr", data=ncre_sl)
            ncr_sl_dr.create_dataset("bbox", data=bbox)


# sort arrays based on increasing orde rin time
ncr_tot_arr = ncr_tot_arr[np.argsort(times)]
times = np.sort(times)

# write data to h5py file
with h5py.File(f"{data_dir}/emission_data_ncr.h5", "a") as pf:
    sim_gr = (
        pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]
    )

    sim_ele_gr = sim_gr["proton"]

    if "ncr_tot" in sim_ele_gr.keys():
        del sim_ele_gr["ncr_tot"]

    ncr_sl_dr = sim_ele_gr.create_group("ncr_tot")
    ncr_sl_dr.create_dataset("ncre_tot", data=ncr_tot_arr)
    ncr_sl_dr.create_dataset("times", data=times.to_value(u.kyr))
