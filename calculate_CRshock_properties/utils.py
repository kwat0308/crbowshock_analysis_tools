import numpy as np
import os
import h5py

from transform_data import get_transformed_data
from bowshock_config import bowshock_config

def get_polar_grid(shape : tuple, bbox, derivs=False):
    '''Convert from 3D uniform grid to 2D polar grid (since we average over azimuthal grid anyways)'''

    low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

    # create meshgrid
    x = np.linspace(low_cor[0], up_cor[0], shape[0])
    y = np.linspace(low_cor[1], up_cor[1], shape[1])
    z = np.linspace(low_cor[2], up_cor[2], shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # convert to polar grid
    R = np.sqrt(X**2 + Y**2 + Z**2)
    ### NOTE: "-" sign needed in front of X since we define x against the bow shock, so we need to flip the coordinate system to take it in the rest frame of the bow shock
    TH = np.arccos(-X / R)
    #PH = np.arctan2(Y, X) # not needed since we average over it later
    dR, dTH = None, None

    if derivs:
        # compute derivatives to get line element in each direction
        dxyz = (up_cor - low_cor) / np.array(shape)
        dR = (X / R) * dxyz[0] + (Y / R) * dxyz[1] + (Z / R) * dxyz[2] 
        PTH = np.sqrt(Z**2 + Y**2)  # polar angle
        dTH = (PTH / R**2) * dxyz[0] + (-Y * X / (PTH * R**2)) * dxyz[1] + (-Z * X / (PTH * R**2)) * dxyz[2]

    return R, TH, dR, dTH

def get_shock_mask(pf : h5py.File, blist, shock="outer", dens_thresh=1e-27):
    '''Get shock mask (outer or termination shock) based on density criterion'''

    dens_data = pf["dens"][()][blist]
    shck_data = get_transformed_data(pf, "shck", blist)

    if shock == "outer":
        return (shck_data > 0) & (dens_data > dens_thresh)
    elif shock == "term":
        return (shck_data > 0) & (dens_data < dens_thresh)
    else:
        raise Exception("shock type can only be 'outer' or 'term'.")
    
def find_common_sims(time : float, output_dir : str):
    '''Find & return all sims that has this timestamp, time in cgs'''

    common_sims = []
    common_sim_files = []

    for sim_name in list(bowshock_config.keys()):
        if sim_name not in os.listdir(output_dir):
            print(f"Simulation {sim_name} not found. Skipping.")
            continue

        # get list of all files
        sim_files = []
        for file in os.listdir(f"{output_dir}/{sim_name}"):
            if file.find("plt") < 0:
                continue
            sim_files.append(file)

        for sim_file in sim_files:
            with h5py.File(os.path.join(output_dir, sim_name, sim_file), "r") as pf:
                curr_time = pf["real scalars"][()][0][1]
                if np.isclose(curr_time, time, rtol=1e-2, atol=1e-2):
                    common_sims.append(sim_name)
                    common_sim_files.append(sim_file)
                    break

    return common_sims, common_sim_files

    


