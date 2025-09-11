#! /bin/bash

# python calculate_ncr_protons.py -S ISM-v30
# python calculate_ncr_protons.py -S ISM-v30-MDiff
# python calculate_ncr_protons.py -S ISM-v30-HDiff
# python calculate_ncr_protons.py -S ISM-v100

# python calculate_ncr_electrons.py -S ISM-v30
# python calculate_ncr_electrons.py -S ISM-v30-MDiff
# python calculate_ncr_electrons.py -S ISM-v30-HDiff
# python calculate_ncr_electrons.py -S ISM-v100

python calculate_gamma_intensity.py -S ISM-v30
python calculate_gamma_intensity.py -S ISM-v30-MDiff
python calculate_gamma_intensity.py -S ISM-v30-HDiff
python calculate_gamma_intensity.py -S ISM-v100

python calculate_synch_intensity.py -S ISM-v30 -ap=4.2 -ae=4.2 --nprocs=1
python calculate_synch_intensity.py -S ISM-v30-MDiff -ap=4.2 -ae=4.2 --nprocs=1
python calculate_synch_intensity.py -S ISM-v30-HDiff -ap=4.2 -ae=4.2 --nprocs=1
python calculate_synch_intensity.py -S ISM-v100 -ap=4.2 -ae=4.2 --nprocs=1
