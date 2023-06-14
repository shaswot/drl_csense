#!/usr/bin/env python3
"""Example of using nbparameterise API to substitute variables in 'batch mode'
"""

from nbclient import execute
import nbformat
from nbparameterise import extract_parameters, parameter_values, replace_definitions

NUM_EXPERIMENTS = 5  # RL algorithms can often be unstable, so we run several experiments (see 

nb_filename = "breakout-train-run.ipynb"
nb = nbformat.read(nb_filename, as_version=4)
orig_parameters = extract_parameters(nb)

for run_no in range(NUM_EXPERIMENTS):
    print(f"Creating notebook for run_{run_no}")

    # Update the parameters
    params = parameter_values(orig_parameters, exp_no=run_no)
    new_nb = replace_definitions(nb, params)

    # Save
    with open(f"{nb_filename.split('.')[0]}_{run_no}.ipynb", 'w') as f:
        nbformat.write(new_nb, f)