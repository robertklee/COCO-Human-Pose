import os
import re
import shutil

import util
from constants import *

# Find all training runs that contain at least one model checkpoint
def find_non_trivial_runs():
    runs = os.listdir(DEFAULT_MODEL_BASE_DIR)

    # run IDs with at least one saved checkpoint
    non_trivial_runs = []

    # Find all run IDs with at least one valid checkpoint
    for run in runs:
        path = os.path.join(DEFAULT_MODEL_BASE_DIR, run)

        # Skip anything that isn't a directory
        if not os.path.isdir(path):
            continue

        # All the files within this training run
        # If this is a training run, it should contain a .csv, .json, and .hdf5
        files = os.listdir(path)

        # Contains at least one valid checkpoint
        contains_checkpoint = False
        for f in files:
            match = re.match(util.MODEL_CHECKPOINT_REGEX, f)

            if match:
                contains_checkpoint = True
                break

        if contains_checkpoint:
            non_trivial_runs.append(run)

    return non_trivial_runs

# Move any runs that do not have at least one checkpoint to a temporary folder for manual verification
def move_trivial_runs(base_dir, non_trivial_runs, verbose=True):
    # Create temporary folder
    DESTINATION_SUBDIR = 'trivial-runs'

    temp_dir = os.path.join(base_dir, DESTINATION_SUBDIR)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print(f'Moving empty training runs in "{base_dir}" to "{temp_dir}". Proceed?')
    input("Press Enter to continue...")

    directories = os.listdir(base_dir)

    directories.remove(DESTINATION_SUBDIR) # Prevent moving directory into itself

    # Move runs with no valid checkpoints into a folder for manual verification
    for d in directories:
        # Skip anything that isn't a directory
        path = os.path.join(base_dir, d)
        if not os.path.isdir(path):
            continue

        if d not in non_trivial_runs:
            # Thus, it's a trivial (empty) run

            src = path
            dest = os.path.join(temp_dir, d)

            if verbose:
                print(f'{src}')
                print(f'>>>> {dest}\n')
            shutil.move(src, dest)

if __name__ == "__main__":
    non_trivial_runs = find_non_trivial_runs()

    move_trivial_runs(DEFAULT_MODEL_BASE_DIR, non_trivial_runs)
    move_trivial_runs(DEFAULT_LOGS_BASE_DIR, non_trivial_runs)
