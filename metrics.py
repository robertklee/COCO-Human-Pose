import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Broken
# https://ai-pool.com/d/keras_iou_implementation
def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.cast(tf.greater(y_pred, .25), dtype=y_true.dtype) # .5 is the threshold
    inter = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(y_true * y_pred, axis=3), axis=2), axis=1)
    union = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return tf.reduce_mean((inter + K.epsilon()) / (union + K.epsilon()))