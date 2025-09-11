'''Plotting in polar coordinates for each model'''

import numpy as np
import os
import json

from transform_data import get_transformed_data
from utils import get_polar_grid, get_shock_mask

import flash_amr_tools
import h5py

from joblib import Parallel, delayed

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--sim_name",
    "-S",
    default="ISM-v30",
    action="store",
    type=str,
    help="Simulation to run."
)

# parser.add_argument(
#     "--sim",
#     "-S",
#     default="ISM-v30",
#     action="store",
#     type=str,
#     help="Simulation to run."
# )

parser.add_argument(
    "--nprocs",
    action="store",
    default=1,
    help="Number of cores to use for parallelisation"
)

args = parser.parse_args()


# flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/cr_bowshock_outputs"
flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs_new"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# initial parameters
field_names = ["dens", "vel", "pres", "mag"]
theta_ranges = [(0,30), (30,60), (60,90), (90,120), (120,150),(150,180)]

# load the bowshock config
bowshock_config = json.load(open("../config/bowshock_config.json", "r"))["config"]

assert args.sim_name in list(bowshock_config.keys()), f"Simulation {args.sim_name} not in bowshock_config.json"

for s, shock in enumerate(["outer", "term"]):

    sim_dir = bowshock_config[args.sim_name]["sim_dir"]

    # get list of all files
    sim_files = []
    for file in os.listdir(f"{flash_output_dir}/{sim_dir}"):
        if file.find("plt") < 0 or file.find("forced") > 0:
            continue
        sim_files.append(file)

    # sort 
    sim_files = sorted(sim_files)

    times = np.zeros(len(sim_files))
    field_polar_vals = np.zeros((len(field_names), len(sim_files), len(theta_ranges)))

    def compute_polar_data(polar_args):
        
        it, ifld, sim_file, field_name = polar_args
        print(sim_file[-4:], field_name)

        blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{flash_output_dir}/{sim_dir}/{sim_file}")

        # open file
        pf = h5py.File(f"{flash_output_dir}/{sim_dir}/{sim_file}")

        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]
        time = pf["real scalars"][()][0][1]  # time in seconds

        shck_mask_data = get_shock_mask(pf, blist, shock=shock)
        field_data = get_transformed_data(pf, field_name, blist)

        pf.close()
    
        # convert to uniform grid
        field_cube = flash_amr_tools.get_cube(field_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
        shck_mask_cube = flash_amr_tools.get_cube(shck_mask_data, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

        del field_data
        del shck_mask_data

        # get polar grid -> in future we should think of a way to get these results without uniform grid :) which will give speed up in calculation
        _, TH, _, _ = get_polar_grid(field_cube.shape, bbox, derivs=False)

        field_polar_val = np.array([
            np.mean(field_cube[(shck_mask_cube == True) & (TH >= np.deg2rad(theta_ranges[i][0])) & (TH < np.deg2rad(theta_ranges[i][1])) & (~np.isnan(field_cube))]) for i in range(len(theta_ranges))
        ])

        del TH
        del field_cube
        del shck_mask_cube

        return it, ifld, time, field_polar_val
    
    sim_argslist = [(it, ifld, sim_files[it], field_names[ifld]) for it, ifld in np.ndindex(len(sim_files), len(field_names))]

    results = Parallel(n_jobs=args.nprocs)(delayed(compute_polar_data)(arg) for arg in sim_argslist)

    for it, ifld, time, field_vals in results:
        times[it] = time
        field_polar_vals[ifld, it, :] = field_vals

    # sort the times
    sorted_indices = np.argsort(times)
    times = times[sorted_indices]
    field_polar_vals = field_polar_vals[:, sorted_indices, :]

    # store the data
    with h5py.File(f"{data_dir}/polar_time_series_{shock}_postshockonly.h5", "a") as pf:
        if args.sim_name in list(pf.keys()):
            sim_gr = pf[args.sim_name]
            del pf[args.sim_name]["times"]
            del pf[args.sim_name]["theta_ranges"]
            sim_gr.create_dataset("times", data=times)
            sim_gr.create_dataset("theta_ranges", data=theta_ranges)
        else:
            sim_gr = pf.create_group(args.sim_name)
            sim_gr.create_dataset("times", data=times)
            sim_gr.create_dataset("theta_ranges", data=theta_ranges)

        for ifld, field in enumerate(field_names):
            if field in list(sim_gr.keys()):
                del sim_gr[field]
            
            sim_gr.create_dataset(field, data=field_polar_vals[ifld,...])

    with h5py.File(f"{data_dir}/polar_time_series_{shock}_postshockonly.h5", "a") as pf:
        print(pf[args.sim_name].keys())