#! /bin/bash

# python calculate_CR_number_density.py -S ISM-v30-LDiff
# python calculate_CR_number_density.py -S ISM-v30-HDiff
# python calculate_CR_number_density.py -S ISM-v60
# python calculate_CR_number_density.py -S ISM-v100

# python calculate_gamma_intensity.py -S ISM-v30-LDiff
# python calculate_gamma_intensity.py -S ISM-v30-HDiff
# python calculate_gamma_intensity.py -S ISM-v60
# python calculate_gamma_intensity.py -S ISM-v100

python calculate_synch_intensity.py -S ISM-v30
python calculate_synch_intensity.py -S ISM-v30-LDiff
python calculate_synch_intensity.py -S ISM-v30-HDiff
python calculate_synch_intensity.py -S ISM-v60
python calculate_synch_intensity.py -S ISM-v100
