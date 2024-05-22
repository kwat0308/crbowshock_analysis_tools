import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt

from transform_data import get_transformed_data
from utils import get_polar_grid, get_shock_mask
from field_config import field_labels, field_lims, field_scales
from bowshock_config import bowshock_config, bow_shock_labels

import flash_amr_tools
import h5py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sim_name", "-S", action="store", type=str, help="name of simulation output.")
parser.add_argument("--shock", action="store", type=str, help="which shock to mask", choices=["outer", "term"], default="outer")

args = parser.parse_args()

fig_dir = "/home/watanabe/Projects/meerkat_bowshocks/figures"
flash_output_dir = "/home/watanabe/Projects/meerkat_bowshocks/outputs"

# initial parameters
sim_name = args.sim_name
sim_label = bowshock_config[args.sim_name]["sim_label"]
shock_label = bow_shock_labels[args.shock]
fields = ["shma", "geff"]
thetas = [[0, 10], [10,30], [30,60], [60,90], [90,120], [120,180]]
theta_colors = ["royalblue", "forestgreen", "indianred", "purple", "goldenrod", "hotpink"]


if not os.path.exists(f"{fig_dir}/{sim_name}"):
    os.mkdir(f"{fig_dir}/{sim_name}")

# get list of all files
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_name}"):
    if file.find("plt") < 0:
        continue
    sim_files.append(file)

# sort 
sim_files = sorted(sim_files)

for field in fields:

    # making directories
    if not os.path.exists(f"{fig_dir}/{sim_name}/timeseries"):
        os.mkdir(f"{fig_dir}/{sim_name}/timeseries")

    if not os.path.exists(f"{fig_dir}/{sim_name}/timeseries/{args.shock}"):
        os.mkdir(f"{fig_dir}/{sim_name}/timeseries/{args.shock}")

    # store time series
    field_aziavgs = np.zeros((len(sim_files), len(thetas)))
    times = np.zeros(len(sim_files))

    for it, sim_file in enumerate(sim_files):
        blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_name}/{sim_file}")

        # open file
        pf = h5py.File(f"{flash_output_dir}/{sim_name}/{sim_file}")

        # skip runs 
        if field not in list(pf.keys()):
            print(f"Field {field} not in simulation file {sim_file}. Continuing.")
            continue

        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]
        
        field_data = get_transformed_data(pf, field, blist)
        times[it] = pf["real scalars"][()][0][1]  # time in seconds
        shck_mask_data = get_shock_mask(pf, blist, shock=args.shock)
        pf.close()

        # convert to uniform grid
        field_cube = flash_amr_tools.get_cube(field_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
        shck_mask_cube = flash_amr_tools.get_cube(shck_mask_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

        # get polar grid -> in future we should think of a way to get these results without uniform grid :) which will give speed up in calculation
        R, TH, _, _ = get_polar_grid(field_cube.shape, bbox)

        for ith, (th_min, th_max) in enumerate(thetas):
            field_aziavgs[it, ith] = np.mean(field_cube[(shck_mask_cube == True) & (TH >= np.deg2rad(th_min)) & (TH < np.deg2rad(th_max))])

    # finally plot it all
    times = (times * u.s).to(u.kyr)
    fig, ax = plt.subplots(figsize=(8,6))

    for ith, (th_min, th_max) in enumerate(thetas):
        ax.plot(times, field_aziavgs[:,ith], label=rf"{th_min}$^\circ$ $\leq$ $\theta$ < {th_max}$^\circ$", color=theta_colors[ith])

    # labels
    ax.set_xlabel("Time / kyr", fontsize=16)
    ax.set_ylabel(field_labels[field], fontsize=16)
    ax.set_title(f"Time series - {sim_label} - {shock_label} Shock", fontsize=20)

    # ylimits consistent
    field_lim = field_lims[field][args.shock] if type(field_lims[field]) == dict else field_lims[field]
    ax.set_ylim(field_lim)
    ax.set_yscale(field_scales[field])

    # ticks
    ax.tick_params(axis='both', direction="in", which='major', size=8, labelsize=18)
    ax.tick_params(axis='both', direction="in", which='minor', size=6, labelsize=16)

    ax.legend(prop={"size":14})

    # save
    fig.savefig(f"{fig_dir}/{sim_name}/timeseries/{args.shock}/{field}_timeseries_{args.shock}shock.png", bbox_inches="tight", dpi=300)

