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
    "geff":r"$\gamma_\mathrm{eff}$",
    "alpha_inj":r"$\alpha_\mathrm{inj}$",
    "mag" : r"$|\vec{B}|$ / $\mu$G",
    "vel" : r"$|\vec{v}|$ / km s$^{-1}$",
    "pres" : "P / $k_B$ / K cm$^{-3}$",
    "einj" : "$\epsilon_\mathrm{CR, inj}$ / erg cm$^{-3}$",
}
field_lims = {
    "dens" : [5e-30, 5e-23],
    # "shma" : {
    #     "outer":[0.0, 15],
    #     "term":[0.0, 35],
    # },
    "shma":[0.8, 50],
    "encr":[1e-15, 1e-11],
    "mgob" : [10, 90],
    "crac": [-0.01, 0.14],
    "geff":[1.4, 1.69],
    "alpha_inj":[3.8, 8.1],
    "mag" : [0.8, 100],
    "vel" : [0.9, 3e4],
    "pres" : [0.9, 5e5],
    "einj" : [1e-30, 1e-14]
}
field_scales = {
    "dens":"log",
    "encr":"log",
    "shma":"log",
    "crac":"linear",
    "mgob":"linear",
    "geff":"linear",
    "alpha_inj":"linear",
    "vel" : "log",
    "mag" : "linear",
    "pres":"log",
    "einj" : "log"
}
field_cmaps = {
    "dens" : cm.Oranges,
    "temp" : cm.coolwarm,
    "pres" : cm.plasma,
    "vel" : cm.YlGnBu,
    "mag" : cm.magma,
    "ener":cm.BuPu,
    "shck" : shck_cmap,
    "shma" : cm.Reds_r,
    "encr" : cm.viridis,
    "entropy" : cm.Blues,
    "mach" : cm.Blues,
    "mgob" : cm.viridis,
    "crac":cm.Reds_r,
    "plasma_beta": cm.PiYG,
    "geff":cm.cividis,
    "einj" : cm.pink
}