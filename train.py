import argparse
import os
import time
from datetime import datetime, timedelta

import tensorflow as tf
from keras import backend as k

from constants import *
from hourglass import HourglassNet

# TODO
# Add command line parameter for learning rate?

def tensorflow_setup():
    print("TensorFlow detected the following GPU(s):")
    tf.test.gpu_device_name()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    ## TODO may need to turn this off for the first few trials to ensure we won't run out of GPU memory mid-training

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))

def process_args():
    argparser = argparse.ArgumentParser(description='Training parameters')
    argparser.add_argument('-m',
                        '--model-save',
                        default=DEFAULT_MODEL_BASE_DIR,
                        help='base directory for saving model weights')
    argparser.add_argument('-e',
                        '--epochs',
                        default=DEFAULT_EPOCHS,
                        type=int,
                        help='number of epochs')
    argparser.add_argument('-b',
                        '--batch',
                        type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help='batch size')
    argparser.add_argument('-hg',
                        '--hourglass',
                        type=int,
                        default=DEFAULT_NUM_HG,
                        help='number of hourglass blocks')
    # Resume model training arguments
    # TODO make a consistent way to generate and retrieve epoch checkpoint filenames
    argparser.add_argument('-r',
                        '--resume',
                        default=False,
                        type=bool,
                        help='resume training')
    argparser.add_argument('--resume-json',
                        default=None,
                        help='Model architecture for re-loading weights to resume training')
    argparser.add_argument('--resume-weights',
                        default=None,
                        help='Model weights file to resume training')
    argparser.add_argument('--resume-epoch',
                        default=None,
                        type=int,
                        help='Epoch to resume training')
    argparser.add_argument('--resume-subdir',
                        default=None,
                        help='Subdirectory containing architecture json and weights')

    # Convert string arguments to appropriate type
    args = argparser.parse_args()

    if args.resume and args.resume_subdir is not None:
        find_resume_json_weights(args)

    return args

def find_resume_json_weights(args):
    enclosing_dir = os.path.join(args.model_save, args.resume_subdir)

    files = os.listdir(enclosing_dir)

    model_jsons         = [f for f in files if (".json" in f and "hpe_hourglass_stacks" in f)]
    model_saved_weights = [f for f in files if (".hdf5" in f and "hpe_epoch{epoch:02d}".format(epoch=args.resume_epoch) in f)]
    
    assert len(model_jsons) > 0
    assert len(model_saved_weights) > 0

    args.resume_json = os.path.join(enclosing_dir, model_jsons[0])
    args.resume_weights = os.path.join(enclosing_dir, model_saved_weights[0])

    print('Found model json: {}'.format(args.resume_json))
    print('Found model weights for epoch {epoch:02d}: {weight_file_name}'.format(epoch=args.resume_epoch, weight_file_name=args.resume_weights))

    assert os.path.exists(args.resume_json)
    assert os.path.exists(args.resume_weights)

if __name__ == "__main__":
    args = process_args()

    print("\n\nSetup start: {}\n".format(time.ctime()))
    setup_start = time.time()

    tensorflow_setup()

    trainingRunTime = datetime.today().strftime('%Y-%m-%d-%Hh-%Mm')

    hgnet = HourglassNet(num_classes=NUM_COCO_KEYPOINTS, num_stacks=args.hourglass, num_channels=NUM_CHANNELS, inres=INPUT_DIM,
                            outres=OUTPUT_DIM)

    training_start = time.time()

    if args.resume:
        print("\n\nResume training start: {}\n".format(time.ctime()))

        hgnet.resume_train(args.batch, args.model_save, args.resume_json, args.resume_weights, args.resume_epoch, args.epochs)
    else:
        hgnet.build_model(show=True)

        print("\n\nTraining start: {}\n".format(time.ctime()))
        print("Hourglass blocks: {:2d}, epochs: {:3d}, batch size: {:2d}".format(args.hourglass, args.epochs, args.batch))

        hgnet.train(args.batch, args.model_save, args.epochs)

    print("\n\nTraining end:   {}\n".format(time.ctime()))

    training_end = time.time()

    setup_time = training_start - setup_start
    training_time = training_end - training_start

    print("Total setup time: {}".format(str(timedelta(seconds=setup_time))))
    print("Total train time: {}".format(str(timedelta(seconds=training_time))))
    print("Hourglass blocks: {:2d}, epochs: {:3d}, batch size: {:2d}".format(args.hourglass, args.epochs, args.batch))
