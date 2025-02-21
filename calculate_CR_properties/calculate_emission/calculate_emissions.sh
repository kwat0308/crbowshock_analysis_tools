#! /bin/bash

python calculate_ncr_protons.py -S ISM-v30
python calculate_ncr_protons.py -S ISM-v30-MDiff
python calculate_ncr_protons.py -S ISM-v30-HDiff
python calculate_ncr_protons.py -S ISM-v100

python calculate_ncr_electrons.py -S ISM-v30
python calculate_ncr_electrons.py -S ISM-v30-MDiff
python calculate_ncr_electrons.py -S ISM-v30-HDiff
python calculate_ncr_electrons.py -S ISM-v100

# python calculate_gamma_intensity.py -S ISM-v30
# python calculate_gamma_intensity.py -S ISM-v30-MDiff
# python calculate_gamma_intensity.py -S ISM-v30-HDiff
# python calculate_gamma_intensity.py -S ISM-v100

python calculate_synch_intensity.py -S ISM-v30
python calculate_synch_intensity.py -S ISM-v30-MDiff
python calculate_synch_intensity.py -S ISM-v30-HDiff
python calculate_synch_intensity.py -S ISM-v100
