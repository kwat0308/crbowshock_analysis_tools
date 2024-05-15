import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.units as u

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from transform_data import get_transformed_data
from field_config import field_cmaps, field_lims, field_labels, shck_color_list, shck_tick_num, shck_ticks

import flash_amr_tools
import h5py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sim_name", "-S", action="store", type=str, help="name of simulation output.")
parser.add_argument("--field", "-F", action="store", type=str, help="field variable")
parser.add_argument("--pos", action="store", type=float, help="Position of slice", default=0.0)
parser.add_argument("--axis", action="store", type=int, help="Axis of slice", default=2)

args = parser.parse_args()

fig_dir = "/home/watanabe/Projects/meerkat_bowshocks/figures"
flash_output_dir = "/home/watanabe/Projects/meerkat_bowshocks/outputs"

# initial parameters
sim_dir = args.sim_name
field = args.field
pos = args.pos
axis = args.axis

if not os.path.exists(f"{fig_dir}/{sim_dir}"):
    os.mkdir(f"{fig_dir}/{sim_dir}")

# get list of all files
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
    if file.find("plt") < 0:
        continue
    sim_files.append(file)

# sort 
sim_files = sorted(sim_files)[::-1]

# loop over each file and get the slice
for sim_file in sim_files:
    print(sim_file)
    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

    # open file
    pf = h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}")
    print(f"List of variables in file: {list(pf.keys())}")
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]
    field_data = get_transformed_data(pf, field, blist)
    pf.close()


    low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

    # extent for imshow
    extent = ([low_cor[0], up_cor[0], low_cor[1], up_cor[1]] * u.cm).to_value(u.pc)

    if not os.path.exists(f"{fig_dir}/{sim_dir}/{field}"):
        os.mkdir(f"{fig_dir}/{sim_dir}/{field}")

    if not os.path.exists(f"{fig_dir}/{sim_dir}/{field}/axis{axis}"):
        os.mkdir(f"{fig_dir}/{sim_dir}/{field}/axis{axis}")

    field_slice = flash_amr_tools.get_slice(field_data, pos=pos, axis=axis, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

    # create imshow of this
    fig, ax = plt.subplots(figsize=(15,8))

    # normaliztion for color
    if field == "shck":
        # define bins and normalize
        bounds = np.array([1, 3, 5, 7]) - 1
        norm = mcolors.BoundaryNorm(bounds, len(shck_color_list))
    elif field == "shma" or field == "mach" or field == "crac" or field == "mgob":
        norm = mcolors.Normalize(vmin=field_lims[field][0], vmax=field_lims[field][1], clip=True)
    else:
        norm = mcolors.LogNorm(vmin=field_lims[field][0], vmax=field_lims[field][1], clip=True)

    
    im = ax.imshow(field_slice.T, norm=norm, cmap=field_cmaps[field], origin="lower", extent=extent)
    ax.scatter(0, 0, s=5.0, color="k")

     # labels
    ax.set_xlabel("$x$ / pc", fontsize=20)
    ax.set_ylabel("$y$ / pc", fontsize=20)
    # ticks
    ax.tick_params(axis='both', direction="in", which='major', size=8, labelsize=18)
    ax.tick_params(axis='both', direction="in", which='minor', size=6, labelsize=16)

    # colorbar
    # make separate axes for each image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # set ticks as text for shck
    if field == "shck":
        cb = fig.colorbar(im, ax=ax, cax=cax, ticks=shck_tick_num)
        cb.ax.set_yticklabels(shck_ticks, rotation=90)
        cb.ax.tick_params(axis='both', which='major', size=0, labelsize=14.5)
        cb.ax.tick_params(axis='both', which='minor', size=0)

    else:
        cb = fig.colorbar(im, ax=ax, cax=cax)
        cb.ax.set_ylabel(field_labels[field], fontsize=20)
        cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
        cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

    # save
    fig.savefig(f"{fig_dir}/{sim_dir}/{field}/axis{axis}/{field}_sl_axis{axis}_{sim_file[-4:]}.png", bbox_inches="tight", dpi=300)

    plt.close(fig)