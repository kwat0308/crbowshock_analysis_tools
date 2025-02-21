import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import h5py
import os
from scipy.interpolate import RegularGridInterpolator, CubicSpline

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from transform_data import get_transformed_data
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
parser.add_argument(
    "--alpha_inj_e",
    "-ae",
    action="store",
    default=4.,
    help="injection spectral index for electrons",
)
parser.add_argument(
    "--alpha_inj",
    "-ap",
    action="store",
    default=4.,
    help="injection spectral index for proton",
)
args = parser.parse_args()

sim_name = args.sim_name
sim_dir = bowshock_config[args.sim_name]["sim_dir"]

with h5py.File(f"{data_dir}/steady_state_spectrum.h5", "r") as f:
    alpha_inj_grid = f["alpha_inj_grid"][()]
    log10_dens_grid = f["log10_dens_grid"][()]
    log10_B_grid = f["log10_B_grid"][()]
    alpha_inj_e_grid = f["alpha_inj_e_grid"][()]
    log10_enratio_grid = f["electron/log10_eneratio_grid"][()]

alpha_inj_e_idx = np.digitize(args.alpha_inj_e, alpha_inj_e_grid, right=True)
alpha_inj_idx = np.digitize(args.alpha_inj, alpha_inj_grid, right=True)
log10_enratio_grid = log10_enratio_grid[alpha_inj_idx, :, alpha_inj_e_idx, :]

# first interpolate the enratio grid
f_log10_eneratio = RegularGridInterpolator(
    (log10_dens_grid, log10_B_grid),
    log10_enratio_grid,
    method="linear",
    fill_value=None,
    bounds_error=False,
)
# f_log10_eneratio = CubicSpline(x=log10_B_grid,
#     y=log10_enratio_grid,
# )


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
encre_mean_arr = np.zeros(len(sim_files)) * u.cm**-3
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
        mag_data = get_transformed_data(pf, "mag", blist)
        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]
        gam = pf["geff"][()][blist]

        times[isim] = pf["real scalars"][()][0][1] * u.s  # time in seconds (this is fine)

    low_cor, up_cor = bbox[0, :, 0], bbox[-1, :, 1]

    # xs_data = (gam + 1) * shma_data**2 / (2 + (gam - 1) * shma_data**2)
    # alpha_inj_data = 3 * xs_data / (xs_data - 1)
    # # force minimal and maximal values
    # alpha_inj_data[alpha_inj_data < 4] = 4  # soft spectral index for background CRs
    # alpha_inj_data[alpha_inj_data >= 10] = 10  # similarly

    # compute the enratio to get CR number density
    mag_data[mag_data < 1e-9] = 1e-9  # constrain lower limit to 1e-9 as per grid limits
    dens_data[dens_data < 1e-30* u.g / u.cm**3] = 1e-30 * u.g / u.cm**3
    enratio_data = (10.0 ** f_log10_eneratio((np.log10(dens_data.value), np.log10(mag_data))))* u.erg

    print(np.min(enratio_data), enratio_data.max(), np.min(np.log10(mag_data)), np.max(np.log10(mag_data)))

    encre_data = (encr_data * dens_data / enratio_data).to(u.cm**-3)

    print(np.any(np.isnan(encre_data)), encre_data.min(), encre_data.max())

    encre_mean_arr[isim] = np.mean(encre_data)

    if isim % 5 == 0 and isim > 0:
        encre_sl = flash_amr_tools.get_slice(
            encre_data,
            pos=args.pos,
            axis=args.axis,
            ref_lvl=ref_lvl,
            bbox=bbox,
            bsize=bsize,
            brefs=brefs,
            bns=bns,
        )
        print(encre_sl.min(), encre_sl.max())

        # write data to h5py file
        with h5py.File(f"{data_dir}/emission_data_ncr.h5", "a") as pf:
            sim_gr = (
                pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]
            )

            sim_ele_gr = sim_gr["electron"] if "electron" in sim_gr.keys() else sim_gr.create_group("electron")
            ncr_sl_dr = sim_gr["electron"]["slices"] if "slices" in sim_ele_gr.keys() else sim_ele_gr.create_group("slices")

            ncr_sl_dr.create_dataset(f"t{isim}0kyr", data=encre_sl)
            ncr_sl_dr.create_dataset(f"t{isim}0kyr_bbox", data=bbox)

# sort arrays based on iencreasing orde rin time
encre_mean_arr = encre_mean_arr[np.argsort(times)]
times = np.sort(times)

# write data to h5py file
with h5py.File(f"{data_dir}/emission_data_ncr.h5", "a") as pf:
    sim_gr = (
        pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]
    )

    sim_ele_gr = sim_gr["electron"]
    sim_ele_gr.create_dataset("times", data=times.to_value(u.kyr))

    if "ncr_mean" in sim_ele_gr.keys():
        del sim_ele_gr["ncr_mean"]

    sim_ele_gr.create_dataset("ncr_mean", data=encre_mean_arr)
    