import matplotlib.cm as cm
import matplotlib.colors as mcolors

# parameters for plotting the shock
shck_color_list = ["blue", "#00cc00", "#ffd700"]
shck_cmap = mcolors.LinearSegmentedColormap.from_list("shock_surface", shck_color_list, len(shck_color_list))
shck_ticks = ["Pre-Shock", "Shock Surface", "Post-Shock"]
shck_tick_num = [1.65, 3.85, 5.65]

field_labels = {
    "dens" : r"$\rho$ / g cm$^{-3}$",
    "shma" : r"$\mathcal{M}_1$",
    "encr" : "$\epsilon_\mathrm{CR}$ / erg cm$^{-3}$",
    "mgob": r"$\theta_B$ / deg",
    "crac": r"$\eta(\mathcal{M}_1, \theta_B)$",
    "geff":r"$\gamma_\mathrm{eff}$"
}
field_lims = {
    "dens" : [5e-29, 5e-21],
    # "shma" : {
    #     "outer":[0.0, 15],
    #     "term":[0.0, 35],
    # },
    "shma":[1.0, 60],
    "encr":[1e-15, 1e-10],
    "mgob" : [-5, 185.0],
    "crac": [0.0, 0.2],
    "geff":[1.5, 1.75],
}
field_scales = {
    "dens":"log",
    "encr":"log",
    "shma":"log",
    "crac":"linear",
    "mgob":"linear",
    "geff":"linear"
}