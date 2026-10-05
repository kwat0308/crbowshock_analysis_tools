import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import astropy.units as u
import astropy.constants as const

import json
import h5py

from synch_emission_functions import f_p_0, f_e_0, mp, me, c, ep_ratio, synch_kernel, crit_freq, ekin

# load the bowshock config
bowshock_config = json.load(open("../../config/bowshock_config.json", "r"))["config"]

# energy parameters
pp_arr = np.logspace(0, 4, 200)
print(
    f"Minimum & maximum of normalised momentum (protons): {(pp_arr.min() * mp * c**2).to(u.GeV):.5f}, {(pp_arr.max() * mp * c**2).to(u.GeV):.5f}"
)

pe_arr = np.logspace(2, 7, 200)
print(
    f"Minimum & maximum of normalised momentum (electrons): {(pe_arr.min() * me * c**2).to(u.GeV):.5f}, {(pe_arr.max() * me * c**2).to(u.GeV):.5f}"
)
Kep, pp0, pe0 = ep_ratio()

# Convert electron momentum to GeV/c
pe_arr_GeV = (pe_arr * me * c**2).to(u.GeV)

# Physical parameters
dens_val = 1e-25 * (u.g / u.cm**3)
r_pos = 5 * u.pc
vel_val = 100 * u.km / u.s

# Simulation parameters
sim_name = "ISM-v40-MDiff"
alpha_inj_vals = [4.0, 4.2, 4.4, 4.6, 4.8]
mag_vals = np.array([0.1, 1, 10]) * u.uG
synch_freq_factor = 0.29

# Line styles for magnetic fields
mag_lss = ['-', '--', '-.']

# Colormap for injection indices
alpha_inj_cmap = cm.plasma
alpha_inj_norm = mcolors.Normalize(vmin=4, vmax=5)


def synch_coeff(Blos):
    """Coefficient in front of synchrotron integral."""
    return (np.sqrt(3) * const.e.gauss.value**3 * Blos.to_value(u.G)) / (me.value * c.value**2)


# Calculate synchrotron kernel * electron spectrum for all combinations
fig, ax = plt.subplots(figsize=(10, 6))

for imag, mag_val in enumerate(mag_vals):
    for ialpha, alpha_inj_val in enumerate(alpha_inj_vals):
        # Calculate electron spectrum
        fe0 = f_e_0(
            alpha_inj_val,
            pe_arr,
            dens_val,
            mag_val,
            Tstar=1e4 * u.K,
            vel=vel_val,
            pmax=bowshock_config[sim_name]["pmax_electr"],
            vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
            r_pos=r_pos,
        )

        f_e = Kep * f_p_0(
            alpha_inj_val,
            pp0,
            dens_val,
            vel_val,
            pmax=bowshock_config[sim_name]["pmax_prot"],
            vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s)
        ) / f_e_0(
            alpha_inj_val,
            pe0,
            dens_val,
            mag_val,
            Tstar=1e4 * u.K,
            vel=vel_val,
            pmax=bowshock_config[sim_name]["pmax_electr"],
            vstar=bowshock_config[sim_name]["vstar"] * (u.km / u.s),
            r_pos=r_pos,
        ) * fe0

        # Calculate critical frequency
        crit_freq_vals = crit_freq(mag_val, pe_arr)

        # Calculate synchrotron frequency
        synch_freq = synch_freq_factor * crit_freq_vals
        synch_kern = synch_coeff(mag_val) * synch_kernel((synch_freq / crit_freq_vals).value)

        # Calculate product
        product = synch_kern * f_e / np.trapz(f_e, pe_arr)  # Normalize the product

        ax.loglog(
            pe_arr_GeV.value,
            product * pe_arr**4,
            color=alpha_inj_cmap(alpha_inj_norm(alpha_inj_val)),
            ls=mag_lss[imag],
            lw=2.0,
        )

ax.set_xlabel(r"$p_e$ [GeV/$c$]", fontsize=16)
ax.set_ylabel(r"$p_e^4 \, f_e(p_e) \, F(\nu / \nu_c)$ [a.u.]", fontsize=16)
ax.set_title(r"Synchrotron Kernel × Electron Spectrum at $\nu$ = $0.29 \, \nu_c$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14, top=True, right=True, direction='in')
ax.tick_params(axis='both', which='minor', labelsize=14, top=True, right=True, direction='in')

# Add colorbar for injection indices
sm = cm.ScalarMappable(cmap=alpha_inj_cmap, norm=alpha_inj_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_ylabel(r"$\alpha_\mathrm{inj}$", fontsize=16)
cbar.ax.tick_params(axis='both', labelsize=14)

# Add legend for magnetic fields
ax.legend(
    handles=[
        mlines.Line2D([], [], label=rf'$B$ = {mag_vals[imag].to_value(u.uG):.1f} $\mu$G', color='k', ls=mag_lss[imag], lw=2.0)
        for imag in range(len(mag_vals))
    ],
    frameon=False,
    fontsize=16,
    loc='lower left'
)

ax.set_ylim(ymin=1e-35, ymax=1e-18)

# fig.tight_layout()
# plt.savefig("/home/kwatanabe/Projects/cr-bowshocks/figures/emission_revised/synch_kern_electron_spec_1d.pdf", dpi=200, bbox_inches='tight')
# plt.show()
