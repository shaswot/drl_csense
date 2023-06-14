#!/usr/bin/env python3
import argparse
from nbclient import execute
import nbformat

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="filename of notebook to execute")
args = parser.parse_args()

# Get notebook filename
nb_filename = args.filename
print(f"Executing {nb_filename}")

# Read Notebook
nb = nbformat.read(nb_filename, as_version=4)

# Execute notebook
execute(nb)

# Save notebook
with open(nb_filename, 'w') as f:
    nbformat.write(nb, f)