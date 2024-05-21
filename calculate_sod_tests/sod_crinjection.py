import numpy as np
import matplotlib.pyplot as plt
import os, sys

from scipy.optimize import fsolve

def pth_ratio(xs, th_gamma, cr_gamma, xi):
    '''Thermal Pressure ratio at shock, with injected CRs'''
    num1 = xi * (xs * (cr_gamma - 1) - (cr_gamma + 1)) * xs**th_gamma
    num2 = -xs * (th_gamma + 1) + (th_gamma - 1)

    denom1 = xi * (xs * (cr_gamma - 1) - (cr_gamma+1))
    denom2 = xs * (th_gamma - 1) - (th_gamma + 1)

    return (num1 + num2) / (denom1 + denom2)

def atwood(xs, th_gamma, cr_gamma, xi):
    '''Atwood number, (rho2 - rho1) / (rho2 + rho1)'''
    ys = pth_ratio(xs, th_gamma, cr_gamma, xi)
    num = ys + xi * (ys - xs**th_gamma) - 1
    denom = ys * th_gamma + xi * (ys - xs**th_gamma) * cr_gamma + th_gamma
    return num / denom

def ptot_ratio(xs, th_gamma, cr_gamma, xi):
    '''Total pressure ratio at region 2 (& 3), between shock & contact discontinuity'''
    ys = pth_ratio(xs, th_gamma, cr_gamma, xi)
    return ys + ((cr_gamma - 1) / (th_gamma - 1)) * xi * (ys - xs**th_gamma)

def pinj_ratio(xs, th_gamma, cr_gamma, xi):
    '''CR pressure injected in region 2, normalized by thermal (=total)pressure in region 1'''
    return (cr_gamma - 1) / (th_gamma - 1) * xi * (pth_ratio(xs, th_gamma, cr_gamma, xi) - xs**th_gamma)


# check if it works

def xs_equation(xs, th_gamma, cr_gamma, xi, ptot1, ptot5, rho1, rho5):
    '''Equation to solve for compression ratio xs'''
    aa = atwood(xs, th_gamma, cr_gamma, xi)
    ptot2 = ptot_ratio(xs, th_gamma, cr_gamma, xi) * ptot1

    # calculate sound speeds
    sound_speed_ratio2 = (ptot5 * rho1) / (ptot1 * rho5)

    term1 = (ptot2 / ptot1 - 1) * (aa / (1 + aa))
    expo_term = 1 - (ptot2 / ptot5) ** ((th_gamma - 1) / (2 * th_gamma))
    term2 = 2 * th_gamma / (th_gamma - 1)**2 * sound_speed_ratio2 * expo_term**2

    return term1 - term2


def get_velocities(xs, init_args):
    '''Get all important velocities'''
    th_gamma, cr_gamma, xi, ptot1, ptot5, rho1, rho5 = init_args 


    ptot2 = ptot_ratio(xs, th_gamma, cr_gamma, xi) * ptot1
    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * ptot5 / rho5)

    vel_post = 2 * c5 / (th_gamma - 1) * (1 - (ptot2 / ptot5) ** ((th_gamma - 1) / (2 * th_gamma)))
    vs = vel_post / (1 - (1 / xs))
    vt = c5 - vel_post / (1 - eta2)

    return vel_post, vs, vt


def velocity_scalar(x, t, init_args):
    '''Analytical velocity'''

    # use initial arguments to compute xs & velocities
    xs = fsolve(xs_equation, 1.0, args=init_args)[0]
    velx2, vs, vt = get_velocities(xs, init_args)

    # now unpack to define the rest of the regions
    th_gamma, cr_gamma, xi, ptot1, ptot5, rho1, rho5 = init_args
    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * ptot5 / rho5)
    
    if (x <= vs * t) and (x > -vt * t):
        return velx2
    elif (x <= -vt * t) and (x > -c5 * t):
        return (1 - eta2) * (x / t + c5)
    else:
        return 0.0

def pressure_scalar(x, t, init_args):
    '''Analytical pressure'''

    # use initial arguments to compute xs & velocities
    xs = fsolve(xs_equation, 1.0, args=init_args)[0]
    velx2, vs, vt = get_velocities(xs, init_args)

    # now unpack to define the rest of the regions
    th_gamma, cr_gamma, xi, ptot1, ptot5, rho1, rho5 = init_args
    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * ptot5 / rho5)

    if x > vs * t:
        return ptot1
    elif (x <= vs * t) and (x > -vt * t):
        return ptot_ratio(xs, th_gamma, cr_gamma, xi) * ptot1
    elif (x <= -vt * t) and (x > -c5 * t):
        return ptot5 * (-eta2 * x / (c5 * t) + (1 - eta2)) ** ((2 * th_gamma) / (th_gamma - 1))
    else:
        return ptot5

def prcr_scalar(x, t, init_args):
    '''CR pressure'''
    # use initial arguments to compute xs & velocities
    xs = fsolve(xs_equation, 1.0, args=init_args)[0]
    velx2, vs, vt = get_velocities(xs, init_args)

    # now unpack to define the rest of the regions
    th_gamma, cr_gamma, xi, ptot1, ptot5, rho1, rho5 = init_args

    if (x <= vs * t) and (x > velx2 * t):
        return pinj_ratio(xs, th_gamma, cr_gamma, xi) * ptot1
    else:
        return 0.0
    
def pth_scalar(x, t, init_args):
    '''Thermal pressure'''
    # use initial arguments to compute xs & velocities
    xs = fsolve(xs_equation, 1.0, args=init_args)[0]
    velx2, vs, vt = get_velocities(xs, init_args)

    # now unpack to define the rest of the regions
    th_gamma, cr_gamma, xi, ptot1, ptot5, rho1, rho5 = init_args
    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * ptot5 / rho5)

    if x > vs * t:
        return ptot1
    elif (x <= vs * t) and (x > velx2 * t):
        return pth_ratio(xs, th_gamma, cr_gamma, xi) * ptot1
    elif (x <= velx2 * t) and (x > -vt * t):
        return ptot_ratio(xs, th_gamma, cr_gamma, xi) * ptot1
    elif (x <= -vt * t) and (x > -c5 * t):
        return ptot5 * (-eta2 * x / (c5 * t) + (1 - eta2)) ** ((2 * th_gamma) / (th_gamma - 1))
    else:
        return ptot5

def density_scalar(x, t, init_args):
    '''Analytical density'''
    # use initial arguments to compute xs & velocities
    xs = fsolve(xs_equation, 1.0, args=init_args)[0]
    velx2, vs, vt = get_velocities(xs, init_args)

    # now unpack to define the rest of the regions
    th_gamma, cr_gamma, xi, ptot1, ptot5, rho1, rho5 = init_args
    eta2 = (th_gamma - 1) / (th_gamma + 1)
    c5 = np.sqrt(th_gamma * ptot5 / rho5)

    if x > vs * t:
        return rho1
    elif (x <= vs * t) and (x > velx2 * t):
        return xs * rho1
    elif (x <= velx2 * t) and (x > -vt * t):
        return rho5 * (ptot_ratio(xs, th_gamma, cr_gamma, xi) * ptot1 / ptot5) ** (1 / th_gamma)
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

def pressure_cr(xarr, t, init_args):
    return np.array([prcr_scalar(x, t, init_args) for x in xarr])

def pressure_thermal(xarr, t, init_args):
    return np.array([pth_scalar(x, t, init_args) for x in xarr])


def shock_loc(xarr, t, init_args, atol=1e-5):
    '''Location of the shock'''
    # use initial arguments to compute xs & velocities
    xs = fsolve(xs_equation, 1.0, args=init_args)[0]
    velx2, vs, vt = get_velocities(xs, init_args)

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
    cr_adiidx = 4/3

    accel_eff = 0.5   # zeta in paper

    cr_thermal_ratio = accel_eff / (1 - accel_eff)   # xi in paper

    # pack this all into tuple of initial conditions
    initial_args = (th_adiidx, cr_adiidx, cr_thermal_ratio, pres_1, pres_5, dens_1, dens_5)

    dens_arr = density(x_arr, time, init_args=initial_args)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_arr + xmid, dens_arr)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    fig.savefig("./test_figures/sod_crinjection_test_dens.png")

    pres_arr = pressure(x_arr, time, init_args=initial_args)
    prcr_arr = pressure_cr(x_arr, time, init_args=initial_args)
    pth_arr = pressure_thermal(x_arr, time, init_args=initial_args)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_arr + xmid, pres_arr, color="k", label="Ptot")
    ax.plot(x_arr + xmid, prcr_arr, color="b", label="Pcr")
    ax.plot(x_arr + xmid, pth_arr, color="r", label = "Pth")
    ax.set_xlabel("x")
    ax.set_ylabel("Pressure")
    ax.legend()
    fig.savefig("./test_figures/sod_crinjection_test_pres.png")

    velx_arr = velocity(x_arr, time, init_args=initial_args)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_arr + xmid, velx_arr)
    ax.set_xlabel("x")
    ax.set_ylabel("Velocity - x")
    fig.savefig("./test_figures/sod_crinjection_test_velx.png")
    ax.set_ylabel("Velocity - x")