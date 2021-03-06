import argparse
import os
import tensorflow as tf
from keras import backend as k
from hourglass import HourglassNet

# TODO change to command line arguments
num_stack = 8

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))

    xnet = HourglassNet(num_classes=16, num_stacks=8, num_channels=256, inres=(256, 256),
                            outres=(64, 64))
    
    xnet.build_model(show=True)