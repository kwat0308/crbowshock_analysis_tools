'''Plotting in polar coordinates for each model'''

import numpy as np
import os
import json

from transform_data import get_transformed_data
from utils import get_polar_grid, get_shock_mask

import flash_amr_tools
import h5py


flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/cr_bowshock_outputs"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# initial parameters
fields = ["dens", "encr", "shma", "crac", "mgob", "geff"]
theta_ranges = [(0, 10), (10,20),(20,30), (30, 45), (45,60), (60,75), (75,90), (90,120), (120,150),(150,180)]

# load the bowshock config
bowshock_config = json.load(open("../config/bowshock_config.json", "r"))["config"]

for s, shock in enumerate(["term"]):

    for sim_name in list(bowshock_config.keys()):
        print(sim_name)

        if sim_name not in ("ISM-v100"):
            continue

        sim_dir = bowshock_config[sim_name]["sim_dir"]

        # get list of all files
        sim_files = []
        for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
            if file.find("plt") < 0 or file.find("forced") > 0:
                continue
            sim_files.append(file)

        # sort 
        sim_files = sorted(sim_files)

        times = np.zeros(len(sim_files))
        field_polar_vals = np.zeros((len(fields), len(sim_files), len(theta_ranges)))

        for it, sim_file in enumerate(sim_files):

            print(sim_file)

            blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}", max_ref_given=7)

            # open file
            pf = h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}")

            ref_lvl = pf["refine level"][()][blist]
            bbox = pf["bounding box"][()][blist]
            bsize = pf["block size"][()][blist]
            times[it] = pf["real scalars"][()][0][1]  # time in seconds

            shck_mask_data = get_shock_mask(pf, blist, shock=shock)

            for ifld, field in enumerate(fields):

                # skip runs 
                if field not in list(pf.keys()):
                    print(f"Field {field} not in simulation file {sim_file}. Continuing.")
                    continue

                field_data = get_transformed_data(pf, field, blist)
            
                # convert to uniform grid
                field_cube = flash_amr_tools.get_cube(field_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
                shck_mask_cube = flash_amr_tools.get_cube(shck_mask_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

                # get polar grid -> in future we should think of a way to get these results without uniform grid :) which will give speed up in calculation
                _, TH, _, _ = get_polar_grid(field_cube.shape, bbox, derivs=False)

                field_polar_vals[ifld,it,:] = np.array([
                    np.mean(field_cube[(shck_mask_cube == True) & (TH >= np.deg2rad(theta_ranges[i][0])) & (TH < np.deg2rad(theta_ranges[i][1]))]) for i in range(len(theta_ranges))
                ])

                del TH
                del field_cube
                del shck_mask_cube
                del field_data

            pf.close()

        # store the data
        with h5py.File(f"{data_dir}/polar_time_series_{shock}.h5", "a") as pf:
            if sim_name in list(pf.keys()):
                sim_gr = pf[sim_name]
            else:
                sim_gr = pf.create_group(sim_name)
                sim_gr.create_dataset("times", data=times)
                sim_gr.create_dataset("theta_ranges", data=theta_ranges)

            for ifld, field in enumerate(fields):
                if field in list(sim_gr.keys()):
                    del sim_gr[field]
                
                sim_gr.create_dataset(field, data=field_polar_vals[ifld,...])