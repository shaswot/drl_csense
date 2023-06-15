#!/usr/bin/env python3
"""Example of using nbparameterise API to substitute variables in 'batch mode'
"""
import argparse
from nbclient import execute
import nbformat
from nbparameterise import extract_parameters, parameter_values, replace_definitions

NUM_EXPERIMENTS = 5  # RL algorithms can often be unstable, so we run several experiments (see 

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="filename of notebook to execute")
args = parser.parse_args()

# Get notebook filename
nb_filename = args.filename
nb = nbformat.read(nb_filename, as_version=4)
orig_parameters = extract_parameters(nb)

for run_no in range(NUM_EXPERIMENTS):
    print(f"Creating notebook for run_{run_no}")

    # Update the parameters
    params = parameter_values(orig_parameters, exp_no=run_no, gpu_id=1+run_no)
    new_nb = replace_definitions(nb, params)

    # Save
    with open(f"{nb_filename.split('.')[0]}-run_{run_no}.ipynb", 'w') as f:
        nbformat.write(new_nb, f)