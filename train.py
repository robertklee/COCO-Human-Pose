import argparse
import os
import time
from datetime import datetime, timedelta

import tensorflow as tf
from keras import backend as k

from constants import *
from hourglass import HourglassNet

# TODO change to command line arguments
num_stack = 8

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
                        default=DEFAULT_MODEL_PATH,
                        help='path to save model')
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

    # Convert string arguments to appropriate type
    args = argparser.parse_args()

    return args

if __name__ == "__main__":
    args = process_args()

    print("\n\nSetup start: {}\n".format(time.ctime()))
    setup_start = time.time()

    tensorflow_setup()

    trainingRunTime = datetime.today().strftime('%Y-%m-%d-%Hh-%Mm')

    hgnet = HourglassNet(num_classes=NUM_COCO_KEYPOINTS, num_stacks=args.hourglass, num_channels=256, inres=(256, 256),
                            outres=(64, 64))

    training_start = time.time()

    if args.resume:
        print("\n\Resume training start: {}\n".format(time.ctime()))

        hgnet.resume_train(args.batch, args.resume_json, args.resume_weights, args.resume_epoch, args.epochs)
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