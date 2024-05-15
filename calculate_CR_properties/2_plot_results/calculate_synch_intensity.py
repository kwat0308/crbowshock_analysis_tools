import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import h5py
import os
from scipy.interpolate import RegularGridInterpolator

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1  import make_axes_locatable

from bowshock_config import bowshock_config

import flash_amr_tools

flash_output_dir = "/home/watanabe/Projects/meerkat_bowshocks/outputs"
fig_dir = "/home/watanabe/Projects/meerkat_bowshocks/figures"
data_dir = "/home/watanabe/Projects/meerkat_bowshocks/meerkat_nrfs/meerkat_nrfs/data/"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sim_name", "-S", action="store", type=str, help="name of sim", choices=list(bowshock_config.keys()))
parser.add_argument("--pos", action="store", type=float, help="Position of slice", default=0.0)
parser.add_argument("--axis", action="store", type=int, help="Axis of slice", default=2)
args = parser.parse_args()

sim_name = args.sim_name
sim_label = bowshock_config[args.sim_name]["sim_label"]
mstar = bowshock_config[args.sim_name]["mstar"] * u.Msun
vstar = bowshock_config[args.sim_name]["vstar"] * (u.km / u.s)
env_name = bowshock_config[args.sim_name]["env_name"]
kappa_10 = bowshock_config[args.sim_name]["kappa_10"] * (u.cm**2 / u.s)

# stannd-off radius
with h5py.File(f"{data_dir}/standoff_radius.h5", "r") as f:

    mstar_grid = f["mstar_grid"][()] * u.Msun
    vstar_grid = f["vstar_grid"][()] * (u.km / u.s)

    R0_grid = f[env_name]["R0_grid"][()] * u.cm

mstar_idx = np.digitize(mstar, mstar_grid, right=True)
vstar_idx = np.digitize(vstar, vstar_grid, right=True)

R0 = R0_grid[mstar_idx, vstar_idx].to(u.pc)
print(f"Stand-off radius: {R0.to(u.pc):.3f}")

with h5py.File(f"{data_dir}/steady_state_spectrum_thr.h5", "r") as f:
    log10_R0_grid = f["log10_R0_grid"][()]
    log10_vel_grid = f["log10_vel_grid"][()]
    log10_kappa10_grid = f["log10_kappa10_grid"][()]
    log10_dens_grid = f["log10_dens_grid"][()]
    alpha_inj_grid = f["alpha_inj_grid"][()]
    log10_Blos_grid = f["synch_emission"]["log10_Blos_grid"][()]
    synch_freq = f["synch_emission"]["synch_freq"][()] * u.GHz
    log10_ensynch_grid_full = f["synch_emission"]["log10_ensynch_grid"][()]
    log10_ene_grid_full = f["synch_emission"]["log10_ene_grid"][()]

log10_ensynch_grid = log10_ensynch_grid_full[:,np.digitize(np.log10(kappa_10.value), log10_kappa10_grid, right=True), np.digitize(np.log10(R0.value), log10_R0_grid, right=True),:,:,:]
# log10_ene_grid = log10_ene_grid_full[:,np.digitize(np.log10(kappa_10.value), log10_kappa10_grid, right=True), np.digitize(np.log10(R0.value), log10_R0_grid, right=True),:,:]

# first interpolate the enratio grid
f_log10_ensynchratio = RegularGridInterpolator((alpha_inj_grid, log10_vel_grid,log10_dens_grid, log10_Blos_grid), log10_ensynch_grid)
# f_log10_eneratio = RegularGridInterpolator((alpha_inj_grid, log10_vel_grid,log10_dens_grid), log10_ene_grid)


if not os.path.exists(f"{fig_dir}/{sim_name}"):
    os.mkdir(f"{fig_dir}/{sim_name}")

# get list of all files
sim_files = []
for file in os.listdir(f"{flash_output_dir}/{sim_name}"):
    if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
        continue
    sim_files.append(file)

# sort 
sim_files = sorted(sim_files)

print(f"List of all simulation files: {sim_files}")

# loop over each file and get the slice
for sim_file in sim_files:
    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_name}/{sim_file}")

    with h5py.File(f"{flash_output_dir}/{sim_name}/{sim_file}", "r") as pf:
        shma_data = pf["shma"][()][blist]
        dens_data = pf["dens"][()][blist] * (u.g / u.cm**3)
        vel_data = [pf["velx"][()][blist], pf["vely"][()][blist], pf["velz"][()][blist]] * (u.cm / u.s)
        mag_data = [pf["magx"][()][blist], pf["magy"][()][blist], pf["magz"][()][blist]]  * u.G
        encr_data = pf["encr"][()][blist] * (u.erg / u.g)
        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]

    low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]
    extent = ([low_cor[0], up_cor[0], low_cor[1], up_cor[1]] * u.cm).to_value(u.pc)

    # calculate injection index
    gam = 5/3

    xs_data = (gam + 1) * shma_data**2 / (2 + (gam - 1) * shma_data**2)
    alpha_inj_data = 3 * xs_data / (xs_data - 1)
    # force minimal and maximal values
    alpha_inj_data[alpha_inj_data < 4] = 10  # soft spectral index for background CRs
    alpha_inj_data[alpha_inj_data >= 10] = 10  # similarly

    # computing velocity magnitude3
    vel_mag_data = np.sqrt(vel_data[0]**2 + vel_data[1]**2 + vel_data[2]**2)

    # first get electron number density
    # eneratio_data = 10.0**f_log10_eneratio((alpha_inj_data, np.log10(vel_mag_data.to_value(u.km/u.s)), np.log10(dens_data.to_value(u.g / u.cm**3)))) * u.GeV
    # necr_data = (encr_data * dens_data / eneratio_data).to(u.cm**-3)
    # del eneratio_data

    # LoS magnetic field  ==  z-axis -> transverse is Bx and By contributions
    Blos_data = np.abs(mag_data[0]) + np.abs(mag_data[1])
    Blos_data[Blos_data < 1e-9 * u.G] = 1e-9 * u.G

    # encr to nele_synch ratio
    ensynchratio_data = 10.0**f_log10_ensynchratio((alpha_inj_data, np.log10(vel_mag_data.to_value(u.km/u.s)), np.log10(dens_data.to_value(u.g / u.cm**3)), np.log10(Blos_data.to_value(u.G)))) * u.GeV

    # nele_synch_data
    nelesynch_data = (encr_data * dens_data / ensynchratio_data).to(u.cm**-3)

    del dens_data
    del encr_data
    del vel_mag_data
    del xs_data
    del ensynchratio_data

    # plot this
    # finally plot the slice
    if not os.path.exists(f"{fig_dir}/{sim_name}/nelesynch"):
        os.mkdir(f"{fig_dir}/{sim_name}/nelesynch")

    if not os.path.exists(f"{fig_dir}/{sim_name}/nelesynch/axis{args.axis}"):
        os.mkdir(f"{fig_dir}/{sim_name}/nelesynch/axis{args.axis}")

    nelesynch_sl = flash_amr_tools.get_slice(nelesynch_data, pos=args.pos, axis=args.axis, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
    print(nelesynch_sl.min(), nelesynch_sl.max())
    # create imshow of this
    fig, ax = plt.subplots(figsize=(8,8))

    norm = mcolors.LogNorm(vmin=1e-41, vmax=1e-16, clip=True)

    im = ax.imshow(nelesynch_sl.T, norm=norm, cmap=cm.RdPu, origin="lower", extent=extent)
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
    cb.ax.set_ylabel(r"$n_\mathrm{e, synch}$ / cm$^{-3}$", fontsize=20)
    cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
    cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

    # save
    fig.savefig(f"{fig_dir}/{sim_name}/nelesynch/axis{args.axis}/nelesynch_sl_axis{args.axis}_{sim_file[-4:]}.png")

    plt.close(fig)


    # plot the synchrotron intensity -> E dN/(dV dt dnu), in units of erg / s / cm^3 / Hz
    # NB: astropy doesnt properly take care of cgs units with magnetic fields, so we convert them to values instead and add units later
    synch_coeff = np.sqrt(3) * Blos_data.to_value(u.G) * const.e.gauss.value**3 / (const.m_e.cgs.value * const.c.cgs.value**2 * synch_freq.to_value(u.Hz))
    synch_intens_data = synch_coeff * (u.erg * u.s**-1 * u.Hz**-1) * nelesynch_data

    # plot this
    # finally plot the slice
    if not os.path.exists(f"{fig_dir}/{sim_name}/synch_intens"):
        os.mkdir(f"{fig_dir}/{sim_name}/synch_intens")

    if not os.path.exists(f"{fig_dir}/{sim_name}/synch_intens/axis{args.axis}"):
        os.mkdir(f"{fig_dir}/{sim_name}/synch_intens/axis{args.axis}")

    synch_intens_sl = flash_amr_tools.get_slice(synch_intens_data, pos=args.pos, axis=args.axis, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

    # create imshow of this
    fig, ax = plt.subplots(figsize=(8,8))

    norm = mcolors.LogNorm(vmin=1e-70, vmax=1e-50, clip=True)

    im = ax.imshow(synch_intens_sl.T, norm=norm, cmap=cm.cividis, origin="lower", extent=extent)
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
    cb.ax.set_ylabel(r"$j_\mathrm{syn}$ / erg s$^{-1}$ Hz$^{-1}$ cm$^{-3}$", fontsize=20)
    cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
    cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

    # save
    fig.savefig(f"{fig_dir}/{sim_name}/synch_intens/axis{args.axis}/synch_intens_sl_axis{args.axis}_{sim_file[-4:]}.png")

    plt.close(fig)
        