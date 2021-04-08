import os
import re
import sys
import unicodedata

from constants import *

MODEL_ARCHITECTURE_JSON_REGEX = f'^{HPE_HOURGLASS_STACKS_PREFIX}.*\.json$'
MODEL_CHECKPOINT_REGEX = f'^{HPE_EPOCH_PREFIX}([\d]+).*\.hdf5$'

def str_to_enum(EnumClass, str):
    try:
        enum_ = EnumClass[str]
    except KeyError:
        return None
    return enum_

def validate_enum(EnumClass, str):
    enum_ = str_to_enum(EnumClass=EnumClass, str=str)

    if enum_ is None:
        print(f'\'{str}\' was not found in possible options for Enum class: {EnumClass.__name__}.')
        print('Available options are:')
        options = [name for name, _ in EnumClass.__members__.items()]
        print(options)
        exit(1)
    return True

def is_highest_epoch_file(model_base_dir, model_subdir, epoch_):
    highest_epoch = get_highest_epoch_file(model_base_dir, model_subdir)

    return epoch_ >= highest_epoch

def get_highest_epoch_file(model_base_dir, model_subdir):
    enclosing_dir = os.path.join(model_base_dir, model_subdir)

    files = os.listdir(enclosing_dir)

    highest_epoch = -1

    for f in files:
        match = re.match(MODEL_CHECKPOINT_REGEX, f)

        if match:
            epoch = int(match.group(1))

            if epoch > highest_epoch:
                highest_epoch = epoch

    return highest_epoch

def get_all_epochs(model_base_dir, model_subdir):
    enclosing_dir = os.path.join(model_base_dir, model_subdir)

    files = os.listdir(enclosing_dir)

    model_saved_weights = {}
    for f in files:
        match = re.match(MODEL_CHECKPOINT_REGEX, f)

        if match:
            epoch = int(match.group(1))
            model_saved_weights[epoch] = f

    return model_saved_weights

def find_resume_json_weights_str(model_base_dir, model_subdir, resume_epoch):
    enclosing_dir = os.path.join(model_base_dir, model_subdir)

    files = os.listdir(enclosing_dir)

    model_jsons         = [f for f in files if re.match(MODEL_ARCHITECTURE_JSON_REGEX, f)]
    model_saved_weights = {}
    for f in files:
        match = re.match(MODEL_CHECKPOINT_REGEX, f)

        if match:
            epoch = int(match.group(1))
            model_saved_weights[epoch] = f

    assert len(model_jsons) > 0, "Subdirectory does not contain any model architecture json files"
    assert len(model_saved_weights) > 0, "Subdirectory does not contain any saved model weights"

    if resume_epoch is None or resume_epoch <= 0:
        resume_epoch = max(k for k, _ in model_saved_weights.items())

        print(f'No epoch number provided. Automatically using largest epoch number {resume_epoch:3d}.')

    resume_json = os.path.join(enclosing_dir, model_jsons[0])
    resume_weights = os.path.join(enclosing_dir, model_saved_weights[resume_epoch])

    print('Found model json:                  {}\n'.format(resume_json))
    print('Found model weights for epoch {epoch:3d}: {weight_file_name}\n'.format(epoch=resume_epoch, weight_file_name=resume_weights))

    assert os.path.exists(resume_json)
    assert os.path.exists(resume_weights)

    return resume_json, resume_weights, resume_epoch

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

# https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
def print_progress_bar(percent, label=None):
    # If a label is provided
    if label is not None and label != '':
        label = label + ': '
    else:
        label = ''

    width = 20 # This width is fixed.
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("%s[%-20s] %d%%" % (label, '='*int(percent*width), int(percent*100)))
    sys.stdout.flush()


if __name__ == "__main__":
    from time import sleep

    for i in range(21):
        print_progress_bar(i/20.0, label="test")
        sleep(0.25)
    print()

    for i in range(21):
        print_progress_bar(i/20.0)
        sleep(0.25)
    print()

    for i in range(21):
        print_progress_bar(i/20.0, label='')
        sleep(0.25)
    print()
