import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import h5py
import os
import json

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1  import make_axes_locatable

import flash_amr_tools

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs"
fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

sigma_pp = 44 * u.mbarn  # total pion cross section
Kp = 0.5  # inelasticity
mol_weight = 1.4  # mean molecular weight
Bhadr_coeff = (const.c.cgs * sigma_pp * Kp / (const.m_p.cgs * mol_weight)).cgs  # hadronic loss coefficient

if not os.path.exists(f"{fig_dir}/gamma"):
    os.mkdir(f"{fig_dir}/gamma")

# load the bowshock config
bowshock_config = json.load(open("../../config/bowshock_config.json", "r"))["config"]

for i, sim_name in enumerate(list(bowshock_config.keys())):

    if sim_name != "ISM-v30":
        continue

    sim_dir = bowshock_config[sim_name]['sim_dir']
    mstar = bowshock_config[sim_name]["mstar"] * u.Msun
    vstar = bowshock_config[sim_name]["vstar"] * (u.km/u.s)
    env_name = bowshock_config[sim_name]["env_name"]
    kappa_10 = bowshock_config[sim_name]["kappa_10"] * (u.cm**2 / u.s)

    # get list of all files
    sim_files = []
    for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
        if file.find("plt") < 0 or file.find("forced") > 0 or file.find("0000") > 0:
            continue
        sim_files.append(file)


    # sort 
    sim_files = sorted(sim_files)

    # append gamma ray luminosity array
    gam_lum_arr = np.zeros(len(sim_files)) * (u.erg / u.s)
    times = np.zeros(len(sim_files)) * u.s

    # loop over each file and get the slice
    for isim, sim_file in enumerate(sim_files):
        blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

        with h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}", "r") as pf:
            dens_data = pf["dens"][()][blist] * (u.g / u.cm**3)
            encr_data = pf["encr"][()][blist] * (u.erg / u.g)
            ref_lvl = pf["refine level"][()][blist]
            bbox = pf["bounding box"][()][blist]
            bsize = pf["block size"][()][blist]

            times[isim] = pf["real scalars"][()][0][1] * u.s # time in seconds

        low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

        # gamma ray luminosity as in Longair eq 15.5
        encr_data = (encr_data * dens_data).to(u.erg / u.pc**3)
        gam_intens_data = (1 / 3 * Bhadr_coeff * dens_data.to(u.g / u.pc**3) * encr_data).to(u.erg / u.s / u.cm**3)

        if isim % 100 == 0 and isim != 0:
            # finally plot the slice
            if not os.path.exists(f"{fig_dir}/gamma/{sim_name}"):
                os.mkdir(f"{fig_dir}/gamma/{sim_name}")

            gam_intens_sl = flash_amr_tools.get_slice(gam_intens_data, pos=0.0, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

            # create imshow of this
            fig, ax = plt.subplots(figsize=(8,8))

            # norm = mcolors.LogNorm(vmin=1e-17, vmax=1e-9, clip=True)
            print(gam_intens_sl.min(), gam_intens_sl.max())
            norm = mcolors.LogNorm()
            extent = ([low_cor[0], up_cor[0], low_cor[1], up_cor[1]] * u.cm).to_value(u.pc)

            im = ax.imshow(gam_intens_sl.T, norm=norm, cmap=cm.GnBu, origin="lower", extent=extent)
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
            cb.ax.set_ylabel(r"$j_\gamma$ / erg s$^{-1}$ cm$^{-3}$", fontsize=20)
            cb.ax.tick_params(axis='both', which='major', size=8, labelsize=18)
            cb.ax.tick_params(axis='both', which='minor', size=6, labelsize=18)

            # save
            fig.savefig(f"{fig_dir}/gamma/{sim_name}/gamma_intens_sl_t{times[isim].to(u.kyr):.1f}.png", bbox_inches="tight", dpi=300)

            plt.close(fig)

        # # compute gamma ray luminosity
        cell_size_data = np.tile(bsize[:,1] / 8.0, reps=(8,8,8)).reshape((bsize.shape[0], 8,8,8)) * u.cm
        gam_lum_arr[isim] = np.sum(gam_intens_data.to(u.erg / u.s / u.pc**3) * cell_size_data.to(u.pc)**3)

    print(gam_lum_arr.min(), gam_lum_arr.max())

    # write data to h5py file
    with h5py.File(f"{data_dir}/emission_data.h5", "a") as pf:
        if sim_name not in pf.keys():
            sim_gr = pf.create_group(sim_name)
        else:
            sim_gr = pf[sim_name]

        if "times" in pf[sim_name].keys():
            del pf[sim_name]["times"]
        sim_gr.create_dataset("times", data=times.to(u.kyr))

        if "gamma_lum" in pf[sim_name].keys():
            del pf[sim_name]["gamma_lum"]
        sim_gr.create_dataset("gamma_lum", data=gam_lum_arr.to(u.erg / u.s))
        