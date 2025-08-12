import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.units as u
import json

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from transform_data import get_transformed_data
from field_config import field_lims, field_labels, field_scales

import flash_amr_tools
import h5py


def get_shock_mask(pf : h5py.File, blist, shock="outer", dens_thresh=1e-27, pres_thresh=2e3):
    '''Get shock mask (outer or termination shock) based on density criterion.
    
    Additional pressure criterion due to artefacts that are somehow produced in the B=Bx run.'''

    dens_data = pf["dens"][()][blist]
    shck_data = get_transformed_data(pf, "shck", blist)
    pres_data = get_transformed_data(pf, "pres", blist)

    if shock == "outer":
        return (shck_data > 0) & (dens_data > dens_thresh) & (pres_data > pres_thresh)
    elif shock == "term":
        return (shck_data > 0) & (dens_data < dens_thresh) & (pres_data > pres_thresh)
    else:
        raise Exception("shock type can only be 'outer' or 'term'.")

flash_output_dir = "/home/kwatanabe/Projects/cr-bowshocks/outputs_new"
data_dir = "/home/kwatanabe/Projects/cr-bowshocks/crbowshock_analysis_tools/data"

# load the bowshock config
bowshock_config = json.load(open("../config/bowshock_config.json", "r"))["config"]

# list of fields we want to plot 1D slice of
fields = [ "vel", "mag", "pth", "prcr"]
# fields = ["prcr"]


for sim_name in list(bowshock_config.keys()):
    # if sim_name != "ISM-v30":
    #     continue
    sim_dir = os.path.join(flash_output_dir, bowshock_config[sim_name]["sim_dir"])

    # get list of all files
    sim_files = []
    for file in os.listdir(sim_dir):
        if file.find("plt") < 0:
            continue
        sim_files.append(file)

    # sort
    sim_files = sorted(sim_files)

    times = np.zeros(len(sim_files))
    mean_fields = np.zeros((len(sim_files), len(fields), 2))

    for isim, sim_file in enumerate(sim_files):
        print(sim_file)

        blist, brefs, bns = flash_amr_tools.get_true_blocks(f"{sim_dir}/{sim_file}")

        # open file to get the coordinate information
        pf = h5py.File(f"{sim_dir}/{sim_file}")
        print(f"List of variables in file: {list(pf.keys())}")
        ref_lvl = pf["refine level"][()][blist]
        bbox = pf["bounding box"][()][blist]
        bsize = pf["block size"][()][blist]
        times[isim] = (pf["real scalars"][()][0][1] * u.s).to_value(u.kyr)  # time in seconds

        # loop over each field since we need a new figure for each of this
        for ifie, field in enumerate(fields):
            if len(field) > 3 and field not in pf.keys():
                continue
            field_data = get_transformed_data(pf, field, blist)

            for ish, shock in enumerate(["outer", "term"]):
                shck_mask_data = get_shock_mask(pf, blist, shock=shock)

                if len(shck_mask_data) < 1:
                    continue
                print(np.mean(field_data[shck_mask_data]))
                mean_fields[isim, ifie, ish] = np.mean(field_data[shck_mask_data])

        pf.close()

    
    # save for each sim
    # store the data
    with h5py.File(f"{data_dir}/time_evo.h5", "a") as pf:
        if sim_name in list(pf.keys()):
            sim_gr = pf[sim_name]
            del pf[sim_name]["times"]
            sim_gr.create_dataset("times", data=times)
        else:
            sim_gr = pf.create_group(sim_name)
            sim_gr.create_dataset("times", data=times)

        for ifld, field in enumerate(fields):
            if field in list(sim_gr.keys()):
                del sim_gr[field]
            
            sim_gr.create_dataset(field, data=mean_fields[:, ifld, :])

    with h5py.File(f"{data_dir}/time_evo.h5", "a") as pf:
        print(pf[sim_name].keys())

        