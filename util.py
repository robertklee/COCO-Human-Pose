import os
import re
import unicodedata

from constants import *

def get_optimizer_enum_from_string(optimizer_str):
    try:
        optimizer = OptimizerType[optimizer_str]
    except KeyError:
        print('OptimizerType was not found in possible options.')
        print('Available options are:')
        available_optimizers = [name for name, member in OptimizerType.__members__.items()]
        print(available_optimizers)
        exit(1)
    return optimizer

def validate_activation(activation):
    try:
        activ = OutputActivation[activation]
    except KeyError:
        print('OutputActivation was not found in possible options.')
        print('Available options are:')
        available_activs = [name for name, member in OutputActivation.__members__.items()]
        print(available_activs)
        exit(1)
    return activ

def find_resume_json_weights_args(args):
    args.model_save, args.resume_subdir, args.resume_epoch = find_resume_json_weights_str(args.model_save, args.resume_subdir, args.resume_epoch)

def find_resume_json_weights_str(model_base_dir, model_subdir, resume_epoch):
    enclosing_dir = os.path.join(model_base_dir, model_subdir)

    files = os.listdir(enclosing_dir)

    model_jsons         = [f for f in files if re.match(f'^{HPE_HOURGLASS_STACKS_PREFIX}.*\.json$', f)]
    model_saved_weights = {}
    for f in files:
        match = re.match(f'^{HPE_EPOCH_PREFIX}([\d]+).*\.hdf5$', f)

        if match:
            epoch = int(match.group(1))
            model_saved_weights[epoch] = f

    assert len(model_jsons) > 0, "Subdirectory does not contain any model architecture json files"
    assert len(model_saved_weights) > 0, "Subdirectory does not contain any saved model weights"

    if resume_epoch is None or resume_epoch <= 0:
        resume_epoch = max(k for k, _ in model_saved_weights.items())

        print(f'No epoch number provided. Automatically using largest epoch number {resume_epoch}.')

    resume_json = os.path.join(enclosing_dir, model_jsons[0])
    resume_weights = os.path.join(enclosing_dir, model_saved_weights[resume_epoch])

    print('Found model json: {}'.format(resume_json))
    print('Found model weights for epoch {epoch:02d}: {weight_file_name}'.format(epoch=resume_epoch, weight_file_name=resume_weights))

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
