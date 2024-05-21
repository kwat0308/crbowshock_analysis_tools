import numpy as np
import matplotlib.pyplot as plt
import os, sys

from scipy.optimize import fsolve


# check if it works

def postshock_pressure(pres2, th_gamma, pres1, pres5, rho1, rho5):
    '''Post-shock pressure computed numerically'''

    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c1 = np.sqrt(th_gamma * pres1 / rho1)
    c5 = np.sqrt(th_gamma * pres5 / rho5)

    exp_term = (th_gamma - 1) / (2 * th_gamma)

    term1 = (pres2 / pres1 - 1) * np.sqrt((1 - eta2) / (th_gamma * (pres2 / pres1 + eta2)))
    term2 = 2 / (th_gamma - 1) * (c5 / c1) * (1 - (pres2 / pres5)**exp_term)

    return term1 - term2

def postshock_density(pres2, init_args):

     # unpacl
    th_gamma, pres1, pres5, rho1, rho5 = init_args 

    eta2 = (th_gamma - 1) / (th_gamma + 1)

    return rho1 * ((pres2 + eta2 * pres1) / (pres1 + eta2 * pres2))

def get_velocities(pres2, init_args):
    '''Get all important velocities'''

    # unpacl
    th_gamma, pres1, pres5, rho1, rho5 = init_args 

    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * pres5 / rho5)
    exp_term = (th_gamma - 1) / (2 * th_gamma)

    vel_post = 2 * c5 / (th_gamma - 1) * (1 - (pres2 / pres5) ** exp_term)

    rho2 = postshock_density(pres2, init_args)
   
    vs = vel_post / (1 - (rho1 / rho2))
    vt = c5 - vel_post / (1 - eta2)

    return vel_post, vs, vt


def velocity_scalar(x, t, init_args):
    '''Analytical velocity'''

    # use initial arguments to compute xs & velocities
    th_gamma, pres1, pres5, rho1, rho5 = init_args 

    pres2 = fsolve(postshock_pressure, pres1, args=init_args)[0]
    velx2, vs, vt = get_velocities(pres2, init_args)

    # now unpack to define the rest of the regions
    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * pres5 / rho5)
    
    if (x <= vs * t) and (x > -vt * t):
        return velx2
    elif (x <= -vt * t) and (x > -c5 * t):
        return (1 - eta2) * (x / t + c5)
    else:
        return 0.0

def pressure_scalar(x, t, init_args):
    '''Analytical pressure'''

     # use initial arguments to compute xs & velocities
    th_gamma, pres1, pres5, rho1, rho5 = init_args 

    pres2 = fsolve(postshock_pressure, pres1, args=init_args)[0]
    velx2, vs, vt = get_velocities(pres2, init_args)

    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * pres5 / rho5)

    if x > vs * t:
        return pres1
    elif (x <= vs * t) and (x > -vt * t):
        return pres2
    elif (x <= -vt * t) and (x > -c5 * t):
        return pres5 * (-eta2 * x / (c5 * t) + (1 - eta2)) ** ((2 * th_gamma) / (th_gamma - 1))
    else:
        return pres5

def density_scalar(x, t, init_args):
    '''Analytical density'''
    # use initial arguments to compute xs & velocities
    th_gamma, pres1, pres5, rho1, rho5 = init_args 

    pres2 = fsolve(postshock_pressure, pres1, args=init_args)[0]
    velx2, vs, vt = get_velocities(pres2, init_args)

    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * pres5 / rho5)

    if x > vs * t:
        return rho1
    elif (x <= vs * t) and (x > velx2 * t):
        return postshock_density(pres2, init_args)
    elif (x <= velx2 * t) and (x > -vt * t):
        return rho5 * (pres2 / pres5) ** (1 / th_gamma)
    elif (x <= -vt * t) and (x > -c5 * t):
        return rho5 * (-eta2 * x / (c5 * t) + (1 - eta2)) ** (2 / (th_gamma - 1))
    else:
        return rho5

# vectorized functions

def velocity(xarr, t, init_args):
    return np.array([velocity_scalar(x, t, init_args) for x in xarr])

def density(xarr, t, init_args):
    return np.array([density_scalar(x, t, init_args) for x in xarr])

def pressure(xarr, t, init_args):
    return np.array([pressure_scalar(x, t, init_args) for x in xarr])

def shock_loc(xarr, t, init_args, atol=1e-5):
    '''Location of the shock'''
    # use initial arguments to compute xs & velocities
    th_gamma, pres1, pres5, rho1, rho5 = init_args 

    pres2 = fsolve(postshock_pressure, pres1, args=init_args)[0]
    velx2, vs, vt = get_velocities(pres2, init_args)

    return xarr[np.abs(xarr - vs * t) < atol]


# test using Pfrommer's configuration
if __name__ == "__main__":
    time = 0.35  # seconds
    xmid = 5
    x_arr = np.linspace(-xmid, xmid, 1000)  # cm

    # initial conditions
    dens_5 = 1.0
    dens_1 = 0.125

    pres_5 = 63.499
    pres_1 = 0.1

    velx_5 = 0.0
    velx_1 = 0.0

    th_adiidx = 5/3

    # pack this all into tuple of initial conditions
    initial_args = (th_adiidx, pres_1, pres_5, dens_1, dens_5)

    dens_arr = density(x_arr, time, init_args=initial_args)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_arr + xmid, dens_arr)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    fig.savefig("./test_figures/sod_test_dens.png")

    pres_arr = pressure(x_arr, time, init_args=initial_args)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_arr + xmid, pres_arr)
    ax.set_xlabel("x")
    ax.set_ylabel("Pressure")
    fig.savefig("./test_figures/sod_test_pres.png")

    velx_arr = velocity(x_arr, time, init_args=initial_args)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_arr + xmid, velx_arr)
    ax.set_xlabel("x")
    ax.set_ylabel("Velocity - x")
    fig.savefig("./test_figures/sod_test_velx.png")
    ax.set_ylabel("Velocity - x")