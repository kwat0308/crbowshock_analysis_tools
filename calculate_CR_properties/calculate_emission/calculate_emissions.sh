#! /bin/bash

# python calculate_ncr_protons.py -S ISM-v30
# python calculate_ncr_protons.py -S ISM-v30-MDiff
# python calculate_ncr_protons.py -S ISM-v30-HDiff
# python calculate_ncr_protons.py -S ISM-v100

# python calculate_ncr_electrons.py -S ISM-v30
# python calculate_ncr_electrons.py -S ISM-v30-MDiff
# python calculate_ncr_electrons.py -S ISM-v30-HDiff
# python calculate_ncr_electrons.py -S ISM-v100

# python calculate_gamma_intensity.py -S ISM-v40-LDiff
# python calculate_gamma_intensity.py -S ISM-v40-MDiff
# python calculate_gamma_intensity.py -S ISM-v40-HDiff
python calculate_gamma_intensity.py -S ISM-v100-MDiff

# python calculate_spectra.py -S ISM-v40-MDiff
# python calculate_synch_intensity.py -S ISM-v40-MDiff --chunk_size=20

# python calculate_spectra.py -S ISM-v40-LDiff
# python calculate_synch_intensity.py -S ISM-v40-LDiff --chunk_size=20

# python calculate_spectra.py -S ISM-v40-HDiff
# python calculate_synch_intensity.py -S ISM-v40-HDiff --chunk_size=20


python calculate_spectra.py -S ISM-v100-MDiff --chunk_size=10
python calculate_synch_intensity.py -S ISM-v100-MDiff --chunk_size=10

# python calculate_cooling_rates.py -S ISM-v40-LDiff --chunk_size=20
# python calculate_cooling_rates.py -S ISM-v40-MDiff --chunk_size=20
# python calculate_cooling_rates.py -S ISM-v40-HDiff --chunk_size=20
# python calculate_cooling_rates.py -S ISM-v100-MDiff --chunk_size=20

# python calculate_synch_emission.py -S ISM-v30 -ae=2.7
# python calculate_synch_emission.py -S ISM-v30-MDiff -ae=2.7
# python calculate_synch_emission.py -S ISM-v30-HDiff -ae=2.7
# python calculate_synch_emission.py -S ISM-v100 -ae=2.7

# python calculate_synch_emission.py -S ISM-v30 -ae=2.2
# python calculate_synch_emission.py -S ISM-v30-MDiff -ae=2.2
# python calculate_synch_emission.py -S ISM-v30-HDiff -ae=2.2
# python calculate_synch_emission.py -S ISM-v100 -ae=2.2
