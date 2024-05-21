import numpy as np
import h5py
import astropy.units as u
import astropy.constants as const

import matplotlib.pyplot as plt

from matplotlib import colormaps
import matplotlib.colors as mcolors
import matplotlib.cm as cm

flash_output_dir = "/home/watanabe/Projects/meerkat_bowshocks/outputs"
fig_dir = "/home/watanabe/Projects/meerkat_bowshocks/figures"
data_dir = "/home/watanabe/Projects/meerkat_bowshocks/meerkat_nrfs/meerkat_nrfs/data/"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", "-E", action="store", type=str, help="name of environment", choices=["ISM", "CMZ"], default="ISM")
args = parser.parse_args()

# read in stand-off radiuss
with h5py.File(f"{data_dir}/standoff_radius.h5", "r") as f:

    mstar_grid = f["mstar_grid"][()] * u.Msun
    vstar_grid = f["vstar_grid"][()] * u.km / u.s

    R0_grid = f[args.env_name]["R0_grid"][()] * u.cm

vstars_list = [30, 60, 100, 150, 200, 250] * (u.km/u.s)

# create color based on normalization
norm = mcolors.Normalize(vmin=10, vmax=300)
cmap = colormaps["bone"]

fig, ax = plt.subplots(figsize=(8,4))

for iv, vstar in enumerate(vstars_list):
    vstar_idx = np.digitize(vstar, vstar_grid, right=True)
    ax.semilogy(mstar_grid, R0_grid[:,vstar_idx].to(u.pc), color=cmap(norm(vstar.value)), ls="-", lw=3.0,)

ax.set_xlabel(r"$M_\star$ / $M_\odot$", fontsize=20)
ax.set_ylabel(r"$R_0$ / pc", fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_ylabel(r"$v_\star$ / km s$^{-1}$", fontsize=18)
cbar.ax.invert_yaxis()
cbar.ax.tick_params(axis='both', which='major', size=6, labelsize=16)
cbar.ax.tick_params(axis='both', which='minor', size=4, labelsize=16)

fig.savefig(f"{fig_dir}/auxilliary/standoff_radius_{args.env_name}.pdf", bbox_inches="tight")