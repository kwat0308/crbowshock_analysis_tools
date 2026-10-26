# crbowshock_analysis_tools

Analysis and plotting code for my simulations of cosmic-ray (CR) acceleration at the bow shocks of runaway massive stars. The simulations are run with FLASH, with CRs injected at the shocks. The code here handles everything around them: setting up the runs, measuring shock properties, computing the non-thermal emission (synchrotron and gamma-ray) from the CR distributions, and making the figures.

The code accompanies the paper *Impact of Cosmic Ray Acceleration on the Early Evolution of Bow Shocks around Massive Runaway Stars* (Watanabe, Walch, Rathjen, Mackey, Nürnberger & Girichidis), [arXiv:2510.11988](https://arxiv.org/abs/2510.11988). If you use it, please cite that paper.

Most of the repository is a mix of Python scripts that do the heavy computation and Jupyter notebooks that plot the results. The scripts are meant to be run from inside their own directory, because they use relative paths to the config and data folders.

## Models

The runs are defined in [config/bowshock_config.json](config/bowshock_config.json). All of them are a 20 M☉ star moving through the ISM.

| Name              | v★ (km/s) | κ(10 GeV) (cm²/s) | CRs |
| ----------------- | --------- | ----------------- | --- |
| `ISM-v40-LDiff`   | 40        | 3×10²⁴            | yes |
| `ISM-v40-MDiff`   | 40        | 3×10²⁵            | yes |
| `ISM-v40-HDiff`   | 40        | 3×10²⁶            | yes |
| `ISM-v100-MDiff`  | 100       | 3×10²⁵            | yes |
| `ISM-v40-MHD`     | 40        | n/a               | no (pure MHD reference) |

The config also stores the plot colour and the maximum proton and electron momenta for each model. Almost every script takes the model name through `-S` / `--sim_name`.

The analysis distinguishes between the two shocks in each bow shock, the outer shock and the termination shock, and many outputs are split that way.

## Layout

```
config/                     model definitions, plot labels/limits for the FLASH fields
data/                       small reference tables (tracked) and larger derived HDF5 files (not tracked)
setup_simulations/          parameter calculations, FLASH setup scripts
calculate_mhd_params/       shock properties and time evolution from the MHD fields
calculate_CRshock_properties/  CR-related shock properties, polar time series
calculate_CR_properties/    CR spectra, cooling, and emission (synchrotron, gamma)
calculate_sod_tests/        Sod shock tube tests, with and without CR injection
plot_results/               generic slice plots and animations of the FLASH output
auxiliary/                  older copies of the slice-plotting helpers
deprecated/                 earlier versions, kept for reference
```

### setup_simulations

`calculate_parameters/` has the notebooks used to work out the stellar properties and simulation parameters, plus a script for the stand-off radius. `scripts/` holds the shell scripts that prepare FLASH run directories for SLURM (`setup_flash.sh`) and run the Sod-tube resolution and Mach-number scans.

### calculate_mhd_params and calculate_CRshock_properties

`get_time_evo.py` and `get_polar_timeseries.py` pick out the shock surfaces and track fields such as density, Mach number, field obliquity and CR energy density over time. The polar version works in polar coordinates around the star. Shocks are identified from the FLASH shock flag, with density and pressure thresholds separating the outer shock from the termination shock (see `get_shock_mask`).

`calculate_polar_timeseries.sh` is the batch wrapper. The `plot_*.ipynb` notebooks read the resulting HDF5 files, and `checks/` holds sanity checks (steady bow shock, stand-off radius, energy budgets).

### calculate_CR_properties

The main part for the emission results, in `calculate_emission/`:

- `synch_emission_functions.py` has the physics: diffusion and acceleration timescales, the proton and electron loss terms (hadronic, Coulomb, adiabatic, synchrotron, inverse Compton on the stellar and background photons), the steady-state spectra, and the synchrotron kernel.
- `calculate_steady_state_grid.py` precomputes steady-state spectra on a grid.
- `calculate_spectra.py`, `calculate_cooling_rates.py`, `calculate_synch_intensity.py` and `calculate_gamma_intensity.py` apply those to the simulation output. Synchrotron intensities are computed at 0.5, 1.5, 3, 5, 10 and 15 GHz.
- `calculate_emissions.sh` runs them for each model. Most lines are commented out, so uncomment whichever you need.

`plot_results/` and `checks/` contain the notebooks for luminosities, spectra, 2D slices and comparisons.

### calculate_sod_tests

`sod.py` is an exact Sod shock tube solver. `sod_crinjection.py` extends it to the case where CRs are injected at the shock, which is what I compared the FLASH CR-injection module against. The notebooks make the comparison plots, including the Brio-Wu and oblique-field (`mgob`) cases.

## Requirements

Python 3 with:

- numpy, scipy, matplotlib
- astropy
- h5py
- joblib, tqdm, psutil
- [flash-amr-tools](https://github.com/kwat0308/flash-amr-tools), which maps the AMR blocks in the FLASH checkpoint files onto uniform grids

```
pip install numpy scipy matplotlib astropy h5py joblib tqdm psutil flash-amr-tools
```

## Running things

A typical emission calculation looks like this:

```
cd calculate_CR_properties/calculate_emission
python calculate_synch_intensity.py -S ISM-v40-MDiff --chunk_size=20
```

`--chunk_size` sets how many blocks are processed at once, so lower it if you run out of memory. The injection spectral indices default to 4.2 and can be changed with `-ae` (electrons) and `-ap` (protons). The results are written to `data/`.

Time evolution of the shocks:

```
cd calculate_CRshock_properties
python get_polar_timeseries.py -S ISM-v40-MDiff
```

Slice plots from the raw FLASH output:

```
cd plot_results
python plot_slice.py -S ISM-v40-MDiff -F dens
```

`-F` is the field name. These are the FLASH variable names or derived ones from `transform_data.py` (`dens`, `temp`, `pres`, `vel`, `mag`, `encr`, `shck`, `shma`, `geff`, `crac`, `mgob`, `einj`). `--pos` and `--axis` set the slice. `produce_plots.sh` and `make_animations.sh` loop over fields and times.

## Before running

- The scripts read the FLASH output and write figures to absolute paths near the top of each file (for example `/home/kwatanabe/Projects/cr-bowshocks/outputs_new`). Change those to match your setup. The simulation output itself is not part of this repository.
- `data/` is partly tracked. The reference tables (`M22_data.h5`, `S14_data.h5`, `source_stellar.h5`, `standoff_radius.h5`) are in git. The larger files (`emission_data_*.h5`, `polar_time_series_*.h5`) are regenerated by the scripts above.
- `calculate_CRshock_properties/bowshock_config.py` is an older model list with the previous naming (`HarpISM` etc.). Use `config/bowshock_config.json` for anything current.

## License

See [LICENSE](LICENSE).
