import matplotlib.cm as cm
import matplotlib.colors as mcolors

# parameters for plotting the shock
shck_color_list = ["blue", "#00cc00", "#ffd700"]
shck_cmap = mcolors.LinearSegmentedColormap.from_list("shock_surface", shck_color_list, len(shck_color_list))
shck_ticks = ["Pre-Shock", "Shock Surface", "Post-Shock"]
shck_tick_num = [1.65, 3.85, 5.65]

field_labels = {
    "dens" : r"$\rho$ / g cm$^{-3}$",
    "temp" : "T / K",
    "pres" : "P / $k_B$ / K cm$^{-3}$",
    "vel" : r"$|\vec{v}|$ / km s$^{-1}$",
    "mag" : r"$|\vec{B}|$ / G",
    "ener":"$\epsilon_\mathrm{tot}$ / erg cm$^{-3}$",
    "shck": None,
    "shma" : r"$\mathcal{M}_1$",
    "encr" : "$\epsilon_\mathrm{CR}$ / erg cm$^{-3}$",
    "mgob": r"$\theta_B$ / deg",
    "crac": r"$\eta(\mathcal{M}_1, \theta_B)$",
    "plasma_beta":r"$\beta = \sqrt{P_\mathrm{th} / P_\mathrm{mag}}$",
    "geff":r"$\gamma_\mathrm{eff}$",
    "alpha_inj":r"$\alpha_\mathrm{inj}$"
}
field_cmaps = {
    "dens" : cm.Oranges,
    "temp" : cm.coolwarm,
    "pres" : cm.plasma,
    "vel" : cm.YlGnBu,
    "mag" : cm.magma,
    "ener":cm.BuPu,
    "shck" : shck_cmap,
    "shma" : cm.Blues,
    "encr" : cm.viridis,
    "entropy" : cm.Blues,
    "mach" : cm.Blues,
    "mgob" : cm.viridis,
    "crac":cm.Reds_r,
    "plasma_beta": cm.PiYG,
    "geff":cm.cividis,
    "alpha_inj":cm.winter,
}
field_scales = {
    "dens" : "log",
    "temp" : "log",
    "pres" : "log",
    "vel" : "log",
    "mag" : "log",
    "ener":"log",
    "shck" : None,
    "shma" : "linear",
    "encr" : "log",
    "entropy" : "log",
    "mach" : "linear",
    "mgob" : "linear",
    "crac": "linear",
    "plasma_beta": "log",
    "geff": "linear",
    "alpha_inj":"linear"
}
field_lims = {
    "dens" : [5e-29, 5e-23],
    "temp" : [7e2, 3e9],
    "pres" : [0.9, 5e5],
    "vel" : [0.9, 3e4],
    "mag" : [7e-9, 3e-4],
    "ener":[1e-12, 1e-9],
    "shck":[0,3],
    "shma" : [0.0, 35],
    "encr":[1e-16, 1e-10],
    "mgob" : [0.0, 90.0],
    "crac": [0.0, 1.0],
    "plasma_beta" : [1e-3, 1e3],
    "geff":[1.3, 1.7],
    "alpha_inj":[4, 8]
}