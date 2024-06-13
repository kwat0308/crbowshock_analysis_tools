'''Plotting in polar coordinates for each model'''

import numpy as np
import os

from transform_data import get_transformed_data
from utils import get_polar_grid, get_shock_mask
from bowshock_config import bowshock_config

import flash_amr_tools
import h5py


flash_output_dir = "/home/watanabe/Projects/meerkat_bowshocks/outputs"
data_dir = "/home/watanabe/Projects/meerkat_bowshocks/crbowshock_analysis_tools/data"

# initial parameters
fields = ["dens", "encr", "shma", "crac", "mgob", "geff"]
thetas = np.linspace(0.1, 179, 10)
theta_bins = np.diff(thetas)
theta_centers = 0.5 * (thetas[1:] + thetas[:-1])  # bin centers of polar angles in degrees

for s, shock in enumerate(["outer", "term"]):

    for sim_name in list(bowshock_config.keys()):

        sim_dir = bowshock_config[sim_name]["sim_dir"]

        # get list of all files
        sim_files = []
        for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
            if file.find("plt") < 0:
                continue
            sim_files.append(file)

        # sort 
        sim_files = sorted(sim_files)

        times = np.zeros(len(sim_files))

        for field in fields:

            field_polar_vals = np.zeros((len(sim_files), len(theta_centers)))

            for it, sim_file in enumerate(sim_files):

                blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

                # open file
                pf = h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}")

                # skip runs 
                if field not in list(pf.keys()):
                    print(f"Field {field} not in simulation file {sim_file}. Continuing.")
                    continue

                ref_lvl = pf["refine level"][()][blist]
                bbox = pf["bounding box"][()][blist]
                bsize = pf["block size"][()][blist]
                times[it] = pf["real scalars"][()][0][1]  # time in seconds
                
                field_data = get_transformed_data(pf, field, blist)
                shck_mask_data = get_shock_mask(pf, blist, shock=shock)
                pf.close()

                # convert to uniform grid
                field_cube = flash_amr_tools.get_cube(field_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
                shck_mask_cube = flash_amr_tools.get_cube(shck_mask_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

                # get polar grid -> in future we should think of a way to get these results without uniform grid :) which will give speed up in calculation
                _, TH, _, _ = get_polar_grid(field_cube.shape, bbox, derivs=False)

                field_polar_vals[it,:] = np.array([
                    np.mean(field_cube[(shck_mask_cube == True) & (TH > theta_centers[i] - theta_bins[i] ) & (TH < theta_centers[i] + theta_bins[i])]) for i in range(len(theta_centers))
                ])

            # store the data
            with h5py.File(f"{data_dir}/polar_time_series_{shock}.h5", "a") as pf:
                if sim_name in list(pf.keys()):
                    sim_gr = pf[sim_name]
                else:
                    sim_gr = pf.create_group(sim_name)
                    sim_gr.create_dataset("times", data=times)
                    sim_gr.create_dataset("theta_centers", data=theta_centers)
                    sim_gr.create_dataset("theta_bins", data=theta_bins)

                if field in list(sim_gr.keys()):
                    del sim_gr[field]
                
                sim_gr.create_dataset(field, data=field_polar_vals)