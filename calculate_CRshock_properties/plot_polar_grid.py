'''Plotting in polar coordinates for each model'''

import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt

from transform_data import get_transformed_data
from utils import get_polar_grid, get_shock_mask, find_common_sims
from field_config import field_labels, field_lims, field_scales
from bowshock_config import bowshock_config, bow_shock_labels

import flash_amr_tools
import h5py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--shock", action="store", type=str, help="which shock to mask", choices=["outer", "term"], default="outer")

args = parser.parse_args()

fig_dir = "/home/watanabe/Projects/meerkat_bowshocks/figures"
flash_output_dir = "/home/watanabe/Projects/meerkat_bowshocks/outputs"

# initial parameters
shock_label = bow_shock_labels[args.shock]
fields = ["dens", "encr", "shma", "crac", "mgob"]
# times = [10, 20, 30, 50, 75, 100, 150] * u.kyr
times = [10, 50, 75, 100] * u.kyr

if not os.path.exists(f"{fig_dir}/polar"):
    os.mkdir(f"{fig_dir}/polar")

if not os.path.exists(f"{fig_dir}/polar/{args.shock}"):
    os.mkdir(f"{fig_dir}/polar/{args.shock}")

for field in fields:

    for time in times:
        # find simulations that all have the times specified
        sim_names, sim_files = find_common_sims(time.to_value(u.s), flash_output_dir)

        if not os.path.exists(f"{fig_dir}/polar/{args.shock}/t{time}"):
            os.mkdir(f"{fig_dir}/polar/{args.shock}/t{time}")

        fig, ax = plt.subplots(figsize=(10,8))

        for im, sim_name in enumerate(sim_names):
            # simulation file for the current timie
            sim_file = sim_files[im]

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
            shck_mask_data = get_shock_mask(pf, blist, shock=args.shock)
            pf.close()

            # convert to uniform grid
            field_cube = flash_amr_tools.get_cube(field_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
            shck_mask_cube = flash_amr_tools.get_cube(shck_mask_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

            # get polar grid -> in future we should think of a way to get these results without uniform grid :) which will give speed up in calculation
            _, TH, _, dTH = get_polar_grid(field_cube.shape, bbox, derivs=True)
            thetas = np.linspace(np.min(TH), np.max(TH), 50)

            field_polar_vals = np.array([
                np.mean(field_cube[(shck_mask_cube == True) & (TH > th - 0.5 * dTH) & (TH < th + 0.5 * dTH)]) for th in thetas
            ])

            ax.plot(np.rad2deg(thetas), field_polar_vals, label=bowshock_config[sim_name]["sim_label"], color=bowshock_config[sim_name]["color"])

        # labels
        ax.set_xlabel(r"$\theta$ / deg", fontsize=16)
        ax.set_ylabel(field_labels[field], fontsize=16)
        ax.set_title(f"Polar distribution - t = {time} - {shock_label} Shock", fontsize=20)

        # ylimits consistent
        field_lim = field_lims[field][args.shock] if type(field_lims[field]) == dict else field_lims[field]
        ax.set_ylim(field_lim)
        ax.set_yscale(field_scales[field])

        # ticks
        ax.tick_params(axis='both', direction="in", which='major', size=8, labelsize=18)
        ax.tick_params(axis='both', direction="in", which='minor', size=6, labelsize=16)

        ax.legend(prop={"size":14})

        # save
        fig.savefig(f"{fig_dir}/polar/{args.shock}/t{time}/{field}_polar_t{time}_{args.shock}shock.png", bbox_inches="tight", dpi=300)

        plt.close()