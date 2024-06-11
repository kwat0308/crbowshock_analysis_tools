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

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--reset", "-r", action="store_true", default=False, help="Reset all synchrotron calculations or not.")

args = parser.parse_args()

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
    synch_freq_grid = f["synch_emission"]["synch_freq_grid"][()] * u.GHz
    log10_ensynch_grid_full = f["synch_emission"]["log10_ensynch_grid"][()]
    log10_ene_grid_full = f["synch_emission"]["log10_ene_grid"][()]


if not os.path.exists(f"{fig_dir}/synch"):
    os.mkdir(f"{fig_dir}/synch")

for i, sim_name in enumerate(list(bowshock_config.keys())):

    # write data to h5py file
    with h5py.File(f"{data_dir}/emission_data.h5", "a") as pf:
        if "synch_lum" in list(pf[sim_name].keys()):
            if args.reset:
                print(f"Reseting calculation for {sim_name}")
                del pf[sim_name]["synch_lum"]
                del pf[sim_name]["synch_freq"]
                del pf[sim_name]["times"]
            else:
                print(f"Synchrotron luminosity for {sim_name} already computed. skipping")
                continue

    sim_dir = bowshock_config[sim_name]['sim_dir']
    mstar = bowshock_config[sim_name]["mstar"] * u.Msun
    vstar = bowshock_config[sim_name]["vstar"] * (u.km/u.s)
    env_name = bowshock_config[sim_name]["env_name"]
    kappa_10 = bowshock_config[sim_name]["kappa_10"] * (u.cm**2 / u.s)

    # stannd-off radius
    with h5py.File(f"{data_dir}/standoff_radius.h5", "r") as f:

        mstar_grid = f["mstar_grid"][()] * u.M_sun
        vstar_grid = f["vstar_grid"][()] * (u.km / u.s)

        R0_grid = f[env_name]["R0_grid"][()] * u.cm

    mstar_idx = np.digitize(mstar, mstar_grid, right=True)
    vstar_idx = np.digitize(vstar, vstar_grid, right=True)

    R0 = R0_grid[mstar_idx, vstar_idx].to(u.pc)
    print(f"Stand-off radius: {R0.to(u.pc):.3f}")

    log10_ensynch_grid = log10_ensynch_grid_full[:,np.digitize(np.log10(kappa_10.value), log10_kappa10_grid, right=True), np.digitize(np.log10(R0.value), log10_R0_grid, right=True),:,:,:,:]

    # first interpolate the enratio grid
    f_log10_ensynchratio = RegularGridInterpolator((alpha_inj_grid, log10_vel_grid,log10_dens_grid, log10_Blos_grid), log10_ensynch_grid)

    # get list of all files
    sim_files = []
    for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
        if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
            continue
        sim_files.append(file)

    # sort 
    sim_files = sorted(sim_files)

    if len(sim_files) > 200:
        sim_files = sim_files[::2]

    # append specific synchrotron luminosity array
    synch_lum_arr = np.zeros((len(sim_files), len(synch_freq_grid))) * (u.erg / u.s / u.Hz)
    times = np.zeros(len(sim_files)) * u.s


    print(f"List of all simulation files: {sim_files}")

    # loop over each file and get the slice
    for isim, sim_file in enumerate(sim_files):
        blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

        with h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r") as pf:
            shma_data = pf["shma"][()][blist]
            dens_data = pf["dens"][()][blist] * (u.g / u.cm**3)
            vel_data = [pf["velx"][()][blist], pf["vely"][()][blist], pf["velz"][()][blist]] * (u.cm / u.s)
            mag_data = [pf["magx"][()][blist], pf["magy"][()][blist], pf["magz"][()][blist]]  * u.G
            encr_data = pf["encr"][()][blist] * (u.erg / u.g)
            ref_lvl = pf["refine level"][()][blist]
            bbox = pf["bounding box"][()][blist]
            bsize = pf["block size"][()][blist]

            times[isim] = pf["real scalars"][()][0][1] * u.s # time in seconds

        low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]
        extent = ([low_cor[0], up_cor[0], low_cor[1], up_cor[1]] * u.cm).to_value(u.pc)

        # calculate injection index
        alpha_inj_data = np.full_like(dens_data.value, fill_value = 4.0)
        # gam = 5/3

        # xs_data = (gam + 1) * shma_data**2 / (2 + (gam - 1) * shma_data**2)
        # alpha_inj_data = 3 * xs_data / (xs_data - 1)
        # # force minimal and maximal values
        # alpha_inj_data[alpha_inj_data < 4] = 4  # soft spectral index for background CRs
        # alpha_inj_data[alpha_inj_data >= 10] = 10  # hard cutoff for soft spectra

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
        nelesynch_data = ((encr_data * dens_data)[...,None] / ensynchratio_data).to(u.cm**-3)

        del dens_data
        del encr_data
        del vel_mag_data
        # del xs_data
        del ensynchratio_data

        # plot the specific synchrotron intensity -> E dN/(dV dt), in units of erg / s / Hz / cm^3
        # NB: astropy doesnt properly take care of cgs units with magnetic fields, so we convert them to values instead and add units later
        synch_coeff = np.sqrt(3) * Blos_data.to_value(u.G) * const.e.gauss.value**3 / (const.m_e.cgs.value * const.c.cgs.value**2)
        synch_intens_data = synch_coeff[...,None] * (u.erg * u.s**-1 * u.Hz**-1) * nelesynch_data

        if isim % 100 == 0 and isim != 0:

            # plot this
            # finally plot the slice
            if not os.path.exists(f"{fig_dir}/synch/{sim_name}"):
                os.mkdir(f"{fig_dir}/synch/{sim_name}")

            for synch_freq in [1.4, 4.5] * u.GHz:
                synch_freq_idx = np.digitize(synch_freq, synch_freq_grid, right=True)
                nelesynch_sl = flash_amr_tools.get_slice(nelesynch_data[...,synch_freq_idx], pos=0.0, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
                print(nelesynch_sl.min(), nelesynch_sl.max())
                # create imshow of this
                fig, ax = plt.subplots(figsize=(8,8))

                norm = mcolors.LogNorm(vmin=1e-20, vmax=1e-9, clip=True)

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
                fig.savefig(f"{fig_dir}/synch/{sim_name}/nelesynch_sl_t{times[isim].to(u.kyr):.1f}_nu{synch_freq.to_value(u.GHz):.1f}GHz.png", bbox_inches="tight", dpi=300)

                plt.close(fig)

                synch_intens_sl = flash_amr_tools.get_slice(synch_intens_data[...,synch_freq_idx] * synch_freq.to_value(u.Hz), pos=0.0, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

                # create imshow of this
                fig, ax = plt.subplots(figsize=(8,8))

                norm = mcolors.LogNorm(vmin=1e-30, vmax=1e-9, clip=True)

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
                cb.ax.set_ylabel(r"$j_\mathrm{syn}$ / erg s$^{-1}$ cm$^{-3}$", fontsize=20)
                cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
                cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

                # save
                fig.savefig(f"{fig_dir}/synch/{sim_name}/synch_intens_sl_t{times[isim].to(u.kyr):.1f}_nu{synch_freq.to_value(u.GHz):.1f}GHz.png", bbox_inches="tight", dpi=300)

                plt.close(fig)

        # # compute synchrotron luminosity
        cell_size_data = np.tile(bsize[:,0] / 8.0, reps=(8,8,8)).reshape((bsize.shape[0], 8,8,8))[...,None] * u.cm
        synch_lum_arr[isim,:] = np.sum(synch_intens_data.to(u.erg / u.s / u.Hz / u.pc**3) * cell_size_data.to(u.pc)**3, axis=(0,1,2,3))

        del synch_intens_data

    print(synch_lum_arr.min(), synch_lum_arr.max())

    # write data to h5py file
    with h5py.File(f"{data_dir}/emission_data.h5", "a") as pf:
        if sim_name not in pf.keys():
            sim_gr = pf.create_group(sim_name)
        else:
            sim_gr = pf[sim_name]

        if "times" not in pf[sim_name].keys():
            sim_gr.create_dataset("times", data=times.to(u.kyr))

        if "synch_lum" not in sim_gr.keys():
            sim_gr.create_dataset("synch_lum", data=synch_lum_arr.to(u.erg / u.s / u.Hz))
            sim_gr.create_dataset("synch_freq", data=synch_freq_grid.to_value(u.GHz))