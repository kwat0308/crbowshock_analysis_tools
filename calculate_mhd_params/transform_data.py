import numpy as np
import h5py

import astropy.constants as const
import astropy.units as u

def get_transformed_data(pf : h5py.File, field : str, blist):
    '''Get data transformed according to the definition'''

    if field == "mag":
        field_data = pf[f"{field}x"][()][blist]**2
        field_data += pf[f"{field}y"][()][blist]**2 
        field_data += pf[f"{field}z"][()][blist]**2 
        field_data = np.sqrt(field_data)
    elif field == "vel":
        field_data = pf[f"{field}x"][()][blist]**2
        field_data += pf[f"{field}y"][()][blist]**2 
        field_data += pf[f"{field}z"][()][blist]**2 
        field_data = np.sqrt(field_data) * u.cm / u.s
        field_data = field_data.to(u.km/u.s)
    elif field == "pres":  # convert to P/kB
        field_data = pf["pres"][()][blist] * (u.g / u.cm / u.s**2) / const.k_B.cgs
        field_data = field_data.to("K/cm**3")
    elif field == "plasma_beta":
        field_data =  8 * np.pi * pf["pres"][()][blist] * (u.g / u.cm / u.s**2)

        field_val_mag = pf[f"magx"][()][blist]**2
        field_val_mag += pf[f"magy"][()][blist]**2 
        field_val_mag += pf[f"magz"][()][blist]**2 
        field_data /= field_val_mag

        del field_val_mag
    elif field == "shck":
        field_data = pf["shck"][()][blist]
        # swap post-shock as 5
        field_data[field_data == 2.0] = 5.0
        # also set everything that is < 1 to nan
        field_data[field_data < 1.0] = np.nan
    elif field == "mgob":
        field_data = np.rad2deg(pf["mgob"][()][blist])

    elif field == "encr":
        field_data = pf["encr"][()][blist] * pf["dens"][()][blist]
    else:
        field_data = pf[field][()][blist]

    return field_data
     

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


