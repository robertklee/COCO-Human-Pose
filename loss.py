from enum import Enum
from util import str_to_enum

import keras
import keras.backend as K
import tensorflow as tf

from constants import LossFunctionOptions

def get_loss_from_string(loss_str):
    loss = str_to_enum(LossFunctionOptions, loss_str)

    if loss is LossFunctionOptions.keras_mse:
        return keras.losses.mean_squared_error
    elif loss is LossFunctionOptions.euclidean_loss:
        return euclidean_loss
    elif loss is LossFunctionOptions.weighted_mse:
        return weighted_mean_squared_error
    elif loss is LossFunctionOptions.focal_loss:
        return focal_loss
    else:
        return None

# categorical_focal_loss and binary_focal_loss credit to
# https://github.com/aldi-dimara/keras-focal-loss/blob/master/focal_loss.py
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss

def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)

        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise

        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise

        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true)*alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1-p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss

# Compatible with tensorflow backend

def focal_loss_fixed(gamma=2., alpha=.25):
	def focal_loss_fixed_func(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed_func

# This isn't ideal since it uses tf operations, but I think the investment required to validate any translation we do
# to Keras backend is lower priority than other backlog items
# The original focal loss was developed for large class imbalance categorical outputs,
# but it looks like this has been adapted for heat maps
def focal_loss(hm_true, hm_pred):
    """
    Computes focal loss for heatmap.
    This function was taken from:
        https://github.com/MioChiu/TF_CenterNet/blob/master/loss.py
    :param hm_true: gt heatmap
    :param hm_pred: predicted heatmap
    :return: loss value
    """
    pos_mask = tf.cast(tf.equal(hm_true, 1.0), dtype=tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1.0), dtype=tf.float32)
    neg_weights = tf.pow(1.0 - hm_true, 4)

    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-5, 1.0 - 1e-5)) * tf.math.pow(1.0 - hm_pred, 2.0) * pos_mask
    neg_loss = (
        -tf.math.log(tf.clip_by_value(1.0 - hm_pred, 1e-5, 1.0 - 1e-5))
        * tf.math.pow(hm_pred, 2.0)
        * neg_weights
        * neg_mask
    )

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return loss

# Shout out to: https://towardsdatascience.com/human-pose-estimation-with-stacked-hourglass-network-and-tensorflow-c4e9f84fd3ce
def weighted_mean_squared_error(y_true, y_pred):
    loss = 0
    # vanilla version
    # loss += tf.math.reduce_mean(tf.math.square(y_true - y_pred))
    # improved version
    weights = tf.cast(y_true > 0, dtype=tf.float32) * 81 + 1
    loss += tf.math.reduce_mean(tf.math.square(y_true - y_pred) * weights)

    return loss

def vanilla_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))
