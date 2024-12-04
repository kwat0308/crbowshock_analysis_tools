import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import h5py
import os
from scipy.interpolate import RegularGridInterpolator

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1  import make_axes_locatable

import flash_amr_tools

import json

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs"
fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# load the bowshock config
bowshock_config = json.load(open("../../config/bowshock_config.json", "r"))["config"]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sim_name", "-S", action="store", type=str, help="name of sim", choices=list(bowshock_config.keys()))
parser.add_argument("--pos", action="store", type=float, help="Position of slice", default=0.0)
parser.add_argument("--axis", action="store", type=int, help="Axis of slice", default=2)
args = parser.parse_args()

sim_name = args.sim_name
sim_dir = bowshock_config[args.sim_name]["sim_dir"]
mstar = bowshock_config[args.sim_name]["mstar"] * u.Msun
vstar = bowshock_config[args.sim_name]["vstar"] * (u.km/u.s)
env_name = bowshock_config[args.sim_name]["env_name"]
kappa_10 = bowshock_config[args.sim_name]["kappa_10"] * (u.cm**2 / u.s)

# read in stand-off radiuss
with h5py.File(f"{data_dir}/standoff_radius.h5", "r") as f:

    mstar_grid = f["mstar_grid"][()] * u.Msun
    vstar_grid = f["vstar_grid"][()] * u.km / u.s

    R0_grid = f[env_name]["R0_grid"][()] * u.cm

mstar_idx = np.digitize(mstar, mstar_grid, right=True)
vstar_idx = np.digitize(vstar, vstar_grid, right=True)

R0 = R0_grid[mstar_idx, vstar_idx].to(u.pc)
print(f"Stand-off radius: {R0.to(u.pc):.3f}")

# read in steady state spectrum
with h5py.File(f"{data_dir}/steady_state_spectrum_thr.h5", "r") as f:
    log10_R0_grid = f["log10_R0_grid"][()]
    log10_vel_grid = f["log10_vel_grid"][()]
    log10_kappa10_grid = f["log10_kappa10_grid"][()]
    log10_dens_grid = f["log10_dens_grid"][()]
    alpha_inj_grid = f["alpha_inj_grid"][()]
    log10_enratio_grid_full = f["log10_enratio_grid"][()]

log10_enratio_grid = log10_enratio_grid_full[:,np.digitize(np.log10(kappa_10.value), log10_kappa10_grid, right=True), np.digitize(np.log10(R0.value), log10_R0_grid, right=True),:,:]

# first interpolate the enratio grid
f_log10enratio = RegularGridInterpolator((alpha_inj_grid, log10_vel_grid,log10_dens_grid), log10_enratio_grid)


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

# loop over each file and get the slice
for isim, sim_file in enumerate(sim_files):
    # if isim > 0:
    #     continue
    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

    with h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r") as pf:
        shma_data = pf["shma"][()][blist]
        dens_data = pf["dens"][()][blist] * (u.g / u.cm**3)
        vel_data = [pf["velx"][()][blist], pf["vely"][()][blist], pf["velz"][()][blist]] 
        encr_data = pf["encr"][()][blist] * (u.erg / u.g)
        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]
        gam = pf["geff"][()][blist]

    low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

    # calculate injection index
    # gam = 5/3

    xs_data = (gam + 1) * shma_data**2 / (2 + (gam - 1) * shma_data**2)
    alpha_inj_data = 3 * xs_data / (xs_data - 1)
    # force minimal and maximal values
    alpha_inj_data[alpha_inj_data < 4] = 4  # soft spectral index for background CRs
    alpha_inj_data[alpha_inj_data >= 10] = 10  # similarly

    # computing velocity magnitude3
    vel_mag_data = np.sqrt(vel_data[0]**2 + vel_data[1]**2 + vel_data[2]**2) * (u.cm / u.s)

    # compute the enratio to get CR number density
    enratio_data = 10.0**f_log10enratio((alpha_inj_data, np.log10(vel_mag_data.to_value(u.km/u.s)), np.log10(dens_data.to_value(u.g/u.cm**3)))) * u.GeV
    ncr_data = (encr_data * dens_data / enratio_data).to(u.cm**-3)

    # finally plot the slice
    if not os.path.exists(f"{fig_dir}/slices/{sim_name}/ncr"):
        os.mkdir(f"{fig_dir}/slices/{sim_name}/ncr")

    ncr_sl = flash_amr_tools.get_slice(ncr_data, pos=args.pos, axis=args.axis, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
    print(ncr_sl.min(), ncr_sl.max())

    # create imshow of this
    fig, ax = plt.subplots(figsize=(8,8))

    norm = mcolors.LogNorm(vmin=1e-14, vmax=1e-8, clip=True)
    extent = ([low_cor[0], up_cor[0], low_cor[1], up_cor[1]] * u.cm).to_value(u.pc)

    im = ax.imshow(ncr_sl.T, norm=norm, cmap=cm.PuBuGn, origin="lower", extent=extent)
    ax.scatter(0, 0, s=5.0, color="k")  # star
 
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
    cb = fig.colorbar(im, ax=ax, cax=cax)
    cb.ax.set_ylabel(r"$n_\mathrm{CR}$ / cm$^{-3}$", fontsize=20)
    cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
    cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

    # save
    fig.savefig(f"{fig_dir}/slices/{sim_name}/ncr/ncr_{sim_file[-4:]}.png", bbox_inches="tight", dpi=300)

    plt.close(fig)
