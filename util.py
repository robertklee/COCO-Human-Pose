import os
import re
import unicodedata

from constants import *


def find_resume_json_weights(args):
    enclosing_dir = os.path.join(args.model_save, args.resume_subdir)

    files = os.listdir(enclosing_dir)

    model_jsons         = [f for f in files if (".json" in f and HPE_HOURGLASS_STACKS_PREFIX in f)]
    model_saved_weights = [f for f in files if (".hdf5" in f and "{prefix}{epoch:02d}".format(prefix=HPE_EPOCH_PREFIX, epoch=args.resume_epoch) in f)]
    
    assert len(model_jsons) > 0, "Subdirectory does not contain any model architecture json files"
    assert len(model_saved_weights) > 0, "Subdirectory does not contain any saved model weights"

    args.resume_json = os.path.join(enclosing_dir, model_jsons[0])
    args.resume_weights = os.path.join(enclosing_dir, model_saved_weights[0])

    print('Found model json: {}'.format(args.resume_json))
    print('Found model weights for epoch {epoch:02d}: {weight_file_name}'.format(epoch=args.resume_epoch, weight_file_name=args.resume_weights))

    assert os.path.exists(args.resume_json)
    assert os.path.exists(args.resume_weights)

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
