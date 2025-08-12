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
import argparse

def find_hdf5_file_from_time(time, sim_dir):
    """
    Find the HDF5 file corresponding to the given time in the simulation directory.
    """
    for file in os.listdir(sim_dir):
        if file.find("hdf5") != -1:
            pf = h5py.File(os.path.join(sim_dir, file), "r")
            time_from_file = (pf['real scalars'][0][1] * u.s).to(u.kyr)
            if np.isclose(time, time_from_file.value, atol=0.1):
                return os.path.join(sim_dir, file)
            
    return None

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs_new"
fig_dir = "/home/kwatanabe/Projects/cr-bowshocks/figures"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# load the bowshock config
bowshock_config = json.load(open("../../config/bowshock_config.json", "r"))["config"]

parser = argparse.ArgumentParser()
parser.add_argument("--sim_name", "-S", action="store", type=str, help="name of sim", choices=list(bowshock_config.keys()))
args = parser.parse_args()

sim_name = args.sim_name

sigma_pp = 44 * u.mbarn  # total pion cross section
Kp = 0.5  # inelasticity
mol_weight = 1.4  # mean molecular weight
Bhadr_coeff = (const.c.cgs * sigma_pp * Kp / (const.m_p.cgs * mol_weight)).cgs  # hadronic loss coefficient

if not os.path.exists(f"{fig_dir}/gamma"):
    os.mkdir(f"{fig_dir}/gamma")

sim_dir = bowshock_config[sim_name]['sim_dir']
mstar = bowshock_config[sim_name]["mstar"] * u.Msun
vstar = bowshock_config[sim_name]["vstar"] * (u.km/u.s)
env_name = bowshock_config[sim_name]["env_name"]
kappa_10 = bowshock_config[sim_name]["kappa_10"] * (u.cm**2 / u.s)

times_to_store = [30, 60, 90, 120, 150]

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
    print(isim)
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

    for t in times_to_store:
        if np.isclose(t, times[isim].to_value(u.kyr), atol=0.1):   # in 30 kyr intervals

            gam_intens_sl = flash_amr_tools.get_slice(gam_intens_data, pos=0.0, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

            # write data to h5py file
            with h5py.File(f"{data_dir}/emission_data_gamma_timecorrected.h5", "a") as pf:
                sim_gr = pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]

                gam_sl_gr = sim_gr["slices"] if "slices" in sim_gr.keys() else sim_gr.create_group("slices")
                
                gam_sl_gr.create_dataset(f"t{t}kyr", data=gam_intens_sl)
                gam_sl_gr.create_dataset(f"t{t}_bbox", data=bbox)

            del gam_intens_sl


    # # compute gamma ray luminosity
    cell_size_data = np.tile(bsize[:,1] / 8.0, reps=(8,8,8)).reshape((bsize.shape[0], 8,8,8)) * u.cm
    gam_lum_arr[isim] = np.sum(gam_intens_data.to(u.erg / u.s / u.pc**3) * cell_size_data.to(u.pc)**3)

print(gam_lum_arr.min(), gam_lum_arr.max())

# sort arrays based on increasing orde rin time
gam_lum_arr = gam_lum_arr[np.argsort(times)]
times = np.sort(times)

# write data to h5py file
with h5py.File(f"{data_dir}/emission_data_gamma_timecorrected.h5", "a") as pf:
    sim_gr = pf.create_group(sim_name) if sim_name not in pf.keys() else pf[sim_name]

    if "times" in pf[sim_name].keys():
        del pf[sim_name]["times"]
    sim_gr.create_dataset("times", data=times.to(u.kyr))

    if "gamma_lum" in pf[sim_name].keys():
        del pf[sim_name]["gamma_lum"]
    sim_gr.create_dataset("gamma_lum", data=gam_lum_arr.to(u.erg / u.s))
    