import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.units as u
import json

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from transform_data import get_transformed_data

import flash_amr_tools
import h5py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sim_name", "-S", action="store", type=str, help="name of simulation output.")
parser.add_argument("--field", "-F", action="store", type=str, default="dens", help="field variable")
parser.add_argument("--pos", action="store", type=float, help="Position of slice", default=0.0)
parser.add_argument("--axis", action="store", type=int, help="Axis of slice", default=2)

args = parser.parse_args()

fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures/slices"
flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs_new"

# initial parameters
sim_name = "BowShock_X3A"
field = args.field
pos = args.pos
axis = args.axis

field_lims = {
    "dens": [1e-22, 1e-18],
    "pres": [1e5, 1e9],
    "temp": [1e3, 1e8],
    "vel": [1e0, 1e4],
}

field_cmaps = {
    "dens": "viridis",
    "pres": "plasma",
    "temp": "inferno",
    "vel": "cividis",
}

field_labels = {
    "dens": r"Density [g cm$^{-3}$]",
    "pres": r"Pressure  / $k_B$ [K cm$^{-3}$]",
    "temp": r"Temperature [K]",
    "vel": r"Velocity [km s$^{-1}$]",
}

sim_dir = os.path.join(flash_output_dir, sim_name)

if not os.path.exists(f"{fig_dir}/{sim_name}"):
    os.mkdir(f"{fig_dir}/{sim_name}")

if not os.path.exists(f"{fig_dir}/{sim_name}/{field}"):
    os.mkdir(f"{fig_dir}/{sim_name}/{field}")

# get list of all files
sim_files = []
for file in os.listdir(sim_dir):
    if file.find("plt") < 0 or file.find('forced') > 0:
        continue
    sim_files.append(file)

# sort 
sim_files = sorted(sim_files)

# truncate to every 10 plot files if we have over 200 plot files
# if len(sim_files) > 200:
#     sim_files = sim_files[::10]

# loop over each file and get the slice
for it, sim_file in enumerate(sim_files):
    print(sim_file)
    # if it == 3 or it == 4:
    #     continue
    # if it > 0:
    #     continue
    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{sim_dir}/{sim_file}")

    # open file
    pf = h5py.File(f"{sim_dir}/{sim_file}")
    print(f"List of variables in file: {list(pf.keys())}")
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]
    field_data = get_transformed_data(pf, field, blist)
    pf.close()

    print(field_data.max(), field_data.min())


    low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

    # extent for imshow
    extent = ([low_cor[0], up_cor[0], low_cor[1], up_cor[1]] * u.cm).to_value(u.pc)

    field_slice = flash_amr_tools.get_slice(field_data, pos=pos, axis=axis, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

    # normaliztion for color
    norm = mcolors.LogNorm(vmin=field_lims[field][0], vmax=field_lims[field][1], clip=True)

    fig, ax = plt.subplots(figsize=(15,8))

    im = ax.imshow(field_slice.T, norm=norm, cmap=field_cmaps[field], origin="lower", extent=extent)
    ax.scatter(0, 0, s=5.0, color="k")

        # labels
    ax.set_xlabel("$x$ / pc", fontsize=20)
    ax.set_ylabel("$y$ / pc", fontsize=20)
    # ticks
    ax.tick_params(axis='both', direction="in", which='major', size=8, labelsize=18)
    ax.tick_params(axis='both', direction="in", which='minor', size=6, labelsize=16)

    ax.set_xlim(-0.025, 0.075)
    ax.set_ylim(-0.025, 0.025)

    # colorbar
    # make separate axes for each image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cb = fig.colorbar(im, ax=ax, cax=cax)
    cb.ax.set_ylabel(field_labels[field], fontsize=20)
    cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
    cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

    # scales
    cb.ax.set_yscale("log")

    # save
    fig.savefig(f"{fig_dir}/{sim_name}/{field}/{field}_{sim_file[-4:]}.png", bbox_inches="tight", dpi=300)

    plt.close(fig)