import numpy as np
import h5py

import astropy.constants as const
import astropy.units as u

def get_transformed_data(pf : h5py.File, field : str, blist):
    '''Get data transformed according to the definition'''

    if field == "vel":
        field_data = pf[f"{field}x"][()][blist]**2
        field_data += pf[f"{field}y"][()][blist]**2 
        field_data += pf[f"{field}z"][()][blist]**2 
        field_data = (np.sqrt(field_data) * (u.cm / u.s)).to(u.km/u.s)
    elif field == "mag":
        field_data = pf[f"{field}x"][()][blist]**2
        field_data += pf[f"{field}y"][()][blist]**2 
        field_data += pf[f"{field}z"][()][blist]**2 
        field_data = np.sqrt(field_data) * u.G
        
    elif field == "pres":  # convert to P/kB
        field_data = pf["pres"][()][blist] * (u.g / u.cm / u.s**2) / const.k_B.cgs
        field_data = field_data.to("K/cm**3")
    elif field == "plasma_beta":
        field_data =  8 * np.pi * pf["pres"][()][blist] * (u.g / u.cm / u.s**2)

        field_val_mag = pf["magx"][()][blist]**2
        field_val_mag += pf["magy"][()][blist]**2 
        field_val_mag += pf["magz"][()][blist]**2 
        field_data /= field_val_mag

        del field_val_mag
    elif field == "shck":
        field_data = pf["shck"][()][blist]
        # swap post-shock as 5
        field_data[field_data == 2.0] = 5.0
        # also set everything that is < 1 to nan
        field_data[field_data < 1.0] = np.nan
    elif field == "shma":
        field_data = pf["shma"][()][blist]
        # also set everything that is < 1 to nan
        field_data[field_data < 1.0] = np.nan
    elif field == "mgob":
        field_data = np.rad2deg(pf["mgob"][()][blist])

    elif field == "encr":
        field_data = pf["encr"][()][blist] * pf["dens"][()][blist]
    elif field == "einj":
        if "einj" in list(pf.keys()):
            field_data = pf["einj"][()][blist] * pf["dens"][()][blist]
        else:
            pass # einj not available in this simulation, so we pass
    elif field == "alpha_inj":
        shma_data = pf["shma"][()][blist]
        geff_data = pf["geff"][()][blist]
        xs_data = (geff_data + 1) * shma_data**2 / (2 + (geff_data - 1) * shma_data**2)
        field_data = 3 * xs_data / (xs_data - 1)
        field_data[shma_data < 2] = np.nan
    elif field == "crac":
        if "crac" in list(pf.keys()):
            field_data = pf["crac"][()][blist]
        else:
            shma_data = pf["shma"][()][blist]
            mgob_data = pf["mgob"][()][blist]

            craccel_shma_data = craccel_machs(shma_data)
            craccel_mgob_data = craccel_mgob(mgob_data)

            field_data = craccel_shma_data * craccel_mgob_data
            field_data[shma_data < 2] = np.nan

        pass
    else:
        field_data = pf[field][()][blist]

    return field_data
     

def craccel_mach(mach, params =  [5.46, -9.78, 4.17, -0.334, 0.57]):
    '''As in Kang & Jones 2007'''
    if mach <= 2:
        return 1.96e-3 * (mach**2 - 1)
    else:
        poly = params[0] * mach**-4 + params[1] * (mach - 1) * mach**-3 + \
                params[2] * (mach - 1)**2 * mach**-2 + \
                params[3] * (mach - 1)**3 * mach**-1 + params[4] * (mach - 1)**4
        return np.sum(poly)
        # return np.sum(params * np.array( [(mach - 1)**n for n in range(0,5)] ) / mach**4)
    
def craccel_mgob(theta_B, theta_crit=np.pi/4, delta_B=np.pi/18):
    '''As in Pais et al 2018'''
    return 0.5 * (np.tanh( (theta_crit - theta_B) / delta_B) + 1)


def craccel_machs(machs, params=  [5.46, -9.78, 4.17, -0.334, 0.57]):
    return np.vectorize(craccel_mach)(machs)
