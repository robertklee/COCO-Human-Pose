import os
import re
import shutil

import util
from constants import *

def find_non_trivial_sessions():
    sessions = os.listdir(DEFAULT_MODEL_BASE_DIR)

    # Session IDs with at least one saved checkpoint
    non_trivial_sessions = []

    # Find all session IDs with at least one valid checkpoint
    for session in sessions:
        path = os.path.join(DEFAULT_MODEL_BASE_DIR, session)

        # Skip anything that isn't a directory
        if not os.path.isdir(path):
            continue

        files = os.listdir(path)

        # Contains at least one valid checkpoint
        contains_checkpoint = False
        for f in files:
            match = re.match(util.MODEL_CHECKPOINT_REGEX, f)

            if match:
                contains_checkpoint = True
                break

        if contains_checkpoint:
            non_trivial_sessions.append(session)

    return non_trivial_sessions

def move_trivial_sessions(base_dir, non_trivial_sessions, verbose=True):
    # Create temporary folder
    DESTINATION_SUBDIR = 'trivial-sessions'

    temp_dir = os.path.join(base_dir, DESTINATION_SUBDIR)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print(f'Moving empty training runs in "{base_dir}" to "{temp_dir}". Proceed?')
    input("Press Enter to continue...")

    directories = os.listdir(base_dir)

    directories.remove(DESTINATION_SUBDIR) # Prevent moving directory into itself

    # Move sessions with no valid checkpoints into a folder for manual verification
    for d in directories:
        # Skip anything that isn't a directory
        if not os.path.isdir(d):
            continue

        if d not in non_trivial_sessions:
            # Empty session

            src = os.path.join(base_dir, d)
            dest = os.path.join(temp_dir, d)

            if verbose:
                print(f'Moving {src} to:')
                print(f'\t\t{dest}\n')
            shutil.move(src, dest)

if __name__ == "__main__":
    non_trivial_sessions = find_non_trivial_sessions()

    move_trivial_sessions(DEFAULT_MODEL_BASE_DIR, non_trivial_sessions)
    move_trivial_sessions(DEFAULT_LOGS_BASE_DIR, non_trivial_sessions)