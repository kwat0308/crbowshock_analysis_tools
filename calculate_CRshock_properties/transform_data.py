import numpy as np
import h5py

import astropy.constants as const
import astropy.units as u

def get_transformed_data(pf : h5py.File, field : str, blist):
    '''Get data transformed according to the definition'''

    if field == "vel" or field == "mag":
        field_data = pf[f"{field}x"][()][blist]**2
        field_data += pf[f"{field}y"][()][blist]**2 
        field_data += pf[f"{field}z"][()][blist]**2 
        field_data = np.sqrt(field_data)
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
    elif field == "alpha_inj":
        shma_data = pf["shma"][()][blist]
        geff_data = pf["geff"][()][blist]
        xs_data = (geff_data + 1) * shma_data**2 / (2 + (geff_data - 1) * shma_data**2)
        field_data = 3 * xs_data / (xs_data - 1)
        field_data[shma_data < 1] = np.nan
    else:
        field_data = pf[field][()][blist]

    return field_data
     


