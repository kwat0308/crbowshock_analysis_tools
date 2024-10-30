import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.units as u
import json

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from transform_data import get_transformed_data
from field_config import field_lims, field_labels, field_scales

import flash_amr_tools
import h5py

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--time",
    "-T",
    action="store",
    type=float,
    default=300,
    help="Time to plot the 1D projection in kyr.",
)

args = parser.parse_args()

fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures/slices"
flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs"

# load the bowshock config
bowshock_config = json.load(open("../config/bowshock_config.json", "r"))["config"]

# initial parameters
time = args.time * u.kyr
sim_idx = int((time / 10 * u.kyr).value)

# list of fields we want to plot 1D slice of
fields = ["dens", "temp", "vel"]

# create imshow of this
fig, ax = plt.subplots(len(fields), 1, figsize=(8, 20))

# legend handles
handles = []

for sim_name in ["ISM-v30", "ISM-v60"]:
    sim_dir = os.path.join(flash_output_dir, bowshock_config[sim_name]["sim_dir"])

    if not os.path.exists(f"{fig_dir}/{sim_name}"):
        os.mkdir(f"{fig_dir}/{sim_name}")

    # get list of all files
    sim_files = []
    for file in os.listdir(sim_dir):
        if file.find("plt") < 0:
            continue
        sim_files.append(file)

    # sort
    sim_file = sorted(sim_files)[sim_idx]
    print(sim_file)

    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{sim_dir}/{sim_file}")

    # open file to get the coordinate information
    pf = h5py.File(f"{sim_dir}/{sim_file}")
    print(f"List of variables in file: {list(pf.keys())}")
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]

    # get size of x and y
    cube_size = np.zeros(3, dtype=int)
    cube_size[0] = int(bns[0]) * 2 ** (brefs[0] - brefs[1] + 3)
    cube_size[1] = int(bns[1]) * 2 ** (brefs[0] - brefs[1] + 3)
    cube_size[2] = int(bns[2]) * 2 ** (brefs[0] - brefs[1] + 3)

    # generate arrays for coordinates
    low_cor, up_cor = bbox[0, :, 0], bbox[-1, :, 1]
    dxs = (up_cor - low_cor) / cube_size
    xarr = (
        np.linspace(low_cor[0], up_cor[0], cube_size[0], endpoint=False) + 0.5 * dxs[0]
    )
    yarr = (
        np.linspace(low_cor[1], up_cor[1], cube_size[1], endpoint=False) + 0.5 * dxs[1]
    )

    # now loop over different fields and plot
    for i, field in enumerate(fields):
        field_data = get_transformed_data(pf, field, blist)

        field_slice = flash_amr_tools.get_slice(
            field_data,
            pos=0.0,
            axis=2,
            ref_lvl=ref_lvl,
            bbox=bbox,
            bsize=bsize,
            brefs=brefs,
            bns=bns,
        )
        # find zero value of y since we cannot slice again
        y_idx = np.digitize(0.0, yarr, right=True)
        field_1dslice = field_slice[:, y_idx]

        # plot
        ax[i].plot(
            (xarr * u.cm).to(u.pc),
            field_1dslice,
            lw=2.0,
            color=bowshock_config[sim_name]["color"],
        )

        # limits
        ax[i].set_xlim(-2, )
        ax[i].set_ylim(field_lims[field])

        # scale
        ax[i].set_yscale(field_scales[field])

        # labels
        ax[i].set_xlabel("$x$ / pc", fontsize=20)
        ax[i].set_ylabel(field_labels[field], fontsize=20)

        # ticks
        ax[i].tick_params(
            axis="both",
            direction="in",
            which="major",
            size=8,
            labelsize=18,
            top=True,
            right=True,
        )
        ax[i].tick_params(
            axis="both",
            direction="in",
            which="minor",
            size=6,
            labelsize=16,
            top=True,
            right=True,
        )

    handles.append(
        mlines.Line2D(
            [0], [0], lw=2.0, color=bowshock_config[sim_name]["color"], label=sim_name
        )
    )

fig.legend(handles=handles)

fig.savefig("./plot.pdf", dpi=200, bbox_inches="tight")

# save
# fig.savefig(f"{fig_dir}/{sim_name}/{field}_{sim_file[-4:]}_1dproj.png", bbox_inches="tight", dpi=300)

# plt.close(fig)
