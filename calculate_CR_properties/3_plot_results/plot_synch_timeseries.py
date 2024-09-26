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
data_dir = "/home/watanabe/Projects/meerkat_bowshocks/crbowshock_analysis_tools/data"


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


if not os.path.exists(f"{fig_dir}/synch"):
    os.mkdir(f"{fig_dir}/synch")

for i, sim_name in enumerate(list(bowshock_config.keys())):

    if i > 0:
        break

    sim_label = bowshock_config[sim_name]["sim_label"]
    mstar = bowshock_config[sim_name]["mstar"] * u.Msun
    vstar = bowshock_config[sim_name]["vstar"] * (u.km/u.s)
    env_name = bowshock_config[sim_name]["env_name"]
    kappa_10 = bowshock_config[sim_name]["kappa_10"] * (u.cm**2 / u.s)

    # stannd-off radius
    with h5py.File(f"{data_dir}/standoff_radius.h5", "r") as f:

        mstar_grid = f["mstar_grid"][()] * u.Msun
        vstar_grid = f["vstar_grid"][()] * (u.km/u.s)

        R0_grid = f[env_name]["R0_grid"][()] * u.cm

    mstar_idx = np.digitize(mstar, mstar_grid, right=True)
    vstar_idx = np.digitize(vstar, vstar_grid, right=True)

    R0 = R0_grid[mstar_idx, vstar_idx].to(u.pc)
    print(f"Stand-off radius: {R0.to(u.pc):.3f}")

    log10_ensynch_grid = log10_ensynch_grid_full[:,np.digitize(np.log10(kappa_10.value), log10_kappa10_grid, right=True), np.digitize(np.log10(R0.value), log10_R0_grid, right=True),:,:,:]

    # first interpolate the enratio grid
    f_log10_ensynchratio = RegularGridInterpolator((alpha_inj_grid, log10_vel_grid,log10_dens_grid, log10_Blos_grid), log10_ensynch_grid)

    # get list of all files
    sim_files = []
    for file in os.listdir(f"{flash_output_dir}/{sim_name}"):
        if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
            continue
        sim_files.append(file)

    # sort 
    sim_files = sorted(sim_files)

    print(f"List of all simulation files: {sim_files}")

    # create grid to store synchrotron emission as time series
    times = np.zeros(len(sim_files)) * u.s

    # get grid size first
    factor = 2  # to increase grid to contain all data

    # get the maximum and minimum refinement levels for the final run such that we have a uniform cube at the maximum refinement level
    blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_name}/{sim_files[-1]}")
    # get also the lower & upper corner of the box so that we can get elongated x-array
    with h5py.File(f"{flash_output_dir}/{sim_name}/{sim_files[-1]}", "r") as pf:
        low_cor, up_cor = pf["bounding box"][()][blist][0,:,0], pf["bounding box"][()][blist][-1,:,1]

    # dimension of elongated grid
    grid_dims = (int(bns[0]) * 2**((brefs[0] - brefs[1]) + 3) * factor, int(bns[1]) * 2**((brefs[0] - brefs[1]) + 3))
    dx = (up_cor[0] * factor - low_cor[0]) / grid_dims[0]
    xsynch_arr = (np.linspace(low_cor[0], up_cor[0] * factor, grid_dims[0], endpoint=False) + 0.5 * dx) * u.cm
    synch_intens_proj_grid = np.zeros(grid_dims) * (u.erg / u.s / u.cm**2 / u.sr)

    extent = ([low_cor[0], up_cor[0] * factor, low_cor[1], up_cor[1]] * u.cm).to_value(u.pc)


    # loop over each file and get the slice
    for isim, sim_file in enumerate(sim_files):

        if isim % 50 != 0:
            continue

        blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_name}/{sim_file}", min_ref_given=6, max_ref_given=6)

        with h5py.File(f"{flash_output_dir}/{sim_name}/{sim_file}", "r") as pf:
            shma_data = pf["shma"][()][blist]
            dens_data = pf["dens"][()][blist] * (u.g / u.cm**3)
            vel_data = [pf["velx"][()][blist], pf["vely"][()][blist], pf["velz"][()][blist]] * (u.cm / u.s)
            mag_data = [pf["magx"][()][blist], pf["magy"][()][blist], pf["magz"][()][blist]]  * u.G
            encr_data = pf["encr"][()][blist] * (u.erg / u.g)
            ref_lvl = pf["refine level"][()][blist]
            bbox = pf["bounding box"][()][blist]
            bsize = pf["block size"][()][blist]

            times[isim] = pf["real scalars"][()][0][1] * u.s # time in seconds
            print(times[isim].to(u.kyr))

        low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

        # calculate injection index
        gam = 5/3

        xs_data = (gam + 1) * shma_data**2 / (2 + (gam - 1) * shma_data**2)
        alpha_inj_data = 3 * xs_data / (xs_data - 1)
        # force minimal and maximal values
        alpha_inj_data[alpha_inj_data < 4] = 6.0  # soft spectral index for background CRs
        alpha_inj_data[alpha_inj_data >= 10] = 10  # similarly

        # alpha_inj_data = np.full(alpha_inj_data.shape, 4.0)

        # computing velocity magnitude3
        vel_mag_data = np.sqrt(vel_data[0]**2 + vel_data[1]**2 + vel_data[2]**2)
        vel_mag_data[vel_mag_data.to(u.km/u.s) >= 5e3*u.km/u.s] = 5e3*u.km/u.s
        vel_mag_data[vel_mag_data.to(u.km/u.s) <= 1e-2*u.km/u.s] = 1e-2*u.km/u.s

        # constrain density data
        dens_data[dens_data <= 1e-30 * u.g/u.cm**3] = 1e-30 * u.g/u.cm**3
        dens_data[dens_data >= 1e-18 * u.g/u.cm**3] = 1e-18 * u.g/u.cm**3

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

        # plot the synchrotron intensity -> E dN/(dV dt dnu), in units of erg / s / cm^3
        # NB: astropy doesnt properly take care of cgs units with magnetic fields, so we convert them to values instead and add units later
        synch_coeff = np.sqrt(3) * Blos_data.to_value(u.G) * const.e.gauss.value**3 / (const.m_e.cgs.value * const.c.cgs.value**2) * synch_freq.to_value(u.Hz)
        synch_intens_data = synch_coeff * (u.erg * u.s**-1) * nelesynch_data

        synch_intens_cube = flash_amr_tools.get_cube(synch_intens_data.value, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns) * (u.erg / u.s / u.cm**3)

        del nelesynch_data
        del synch_intens_data

        # now find where vstar * time is inside elongated x array
        x_shiftidx = np.digitize((vstar * times[isim]).to(u.pc), xsynch_arr.to(u.pc), right=True)
        print(x_shiftidx)

        # get z-coordinate at z=0 so that we only integrate from the origin instead of the whole domain
        dz = (up_cor[2] - low_cor[2]) / synch_intens_cube.shape[2]
        z_arr = ((np.linspace(low_cor[2], up_cor[2], synch_intens_cube.shape[2], endpoint=False) + 0.5 * dz) * u.cm)
        z0_idx = np.digitize(0.0 * u.pc, z_arr.to(u.pc), right=True)

        # add such that we shift based on where star is in lab frame
        synch_intes_proj = np.trapz(y=synch_intens_cube[...,z0_idx:], x=z_arr[z0_idx:], axis=2) / (4 * np.pi * u.sr)
        synch_intens_proj_grid[x_shiftidx:x_shiftidx+synch_intens_cube.shape[0],:] += synch_intes_proj
        synch_intens_proj_grid[:x_shiftidx,:] += np.min(synch_intes_proj)
        synch_intens_proj_grid[x_shiftidx+synch_intens_cube.shape[0]:,:] += np.min(synch_intes_proj)
        
        del synch_intens_cube
        del synch_intes_proj

    print(synch_intens_proj_grid.min(), synch_intens_proj_grid.max())

    fig, ax = plt.subplots(figsize=(10,8))

    # norm = mcolors.LogNorm(vmin=5e-14, vmax=8e-12, clip=True)
    # im = ax.imshow(synch_intens_proj_grid.value.T, cmap=cm.inferno, alpha=1.0, interpolation="none", extent=extent, origin="lower", norm=norm)
    norm = mcolors.Normalize(0.0, 1.0, clip=True)
    im = ax.imshow(synch_intens_proj_grid.value.T / np.max(synch_intens_proj_grid.value.T), cmap=cm.viridis, alpha=1.0, interpolation="none", extent=extent, origin="lower", norm=norm)
    del synch_intens_proj_grid
        
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
    # cb.ax.set_ylabel(r"$I_\mathrm{syn}$ / erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$", fontsize=20)
    cb.ax.set_ylabel(r"$I_\mathrm{syn}$ / $I_\mathrm{syn, 0}$", fontsize=20)
    cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
    cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

    # save
    fig.savefig(f"{fig_dir}/synch/{sim_name}/synch_intens_proj_alpha6_relnorm.png", bbox_inches="tight", dpi=300)