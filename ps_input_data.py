from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import scipy
from scipy import ndimage
import matplotlib
matplotlib.use('TkAgg')


feature_shape = [2000, 199, 1]

num_labels = 88
num_features = feature_shape[0] * feature_shape[1] * feature_shape[2]


def tfrecord_parser(serialized_example):
    """Parses a single tf.Example into spectrogram and label tensors."""
    example = tf.parse_single_example(
        serialized_example,
        features={"spec": tf.FixedLenFeature([num_features], tf.float32),
                  "label": tf.FixedLenFeature([num_labels], tf.int64)})
    features = tf.cast(example['spec'], tf.float32)
    # Reshape spec data into the original shape
    features = tf.reshape(features, feature_shape)
    label = tf.cast(example["label"], tf.int64)
    return features, label


def tfrecord_non_overlap_parser(serialized_example):
    """Parses a single tf.Example into spectrogram and label tensors."""
    example = tf.parse_single_example(
        serialized_example,
        features={"spec": tf.FixedLenFeature([num_features], tf.float32),
                  "label": tf.FixedLenFeature([2000*num_labels], tf.int64)})
    features = tf.cast(example['spec'], tf.float32)
    # Reshape spec data into the original shape
    features = tf.reshape(features, feature_shape)
    label = tf.cast(example["label"], tf.int64)
    label = tf.reshape(label, [2000, 88])
    return features, label


def tfrecord_train_input_fn(filepath, batch_size, num_epochs):
    # estimators optimize to cpu input pipeline on their own
    # with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset(filepath)

    # Map the parser over dataset, and batch results by up to batch_size
    # dataset = dataset.shuffle(100000)
    # dataset = dataset.repeat(num_epochs)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(500, num_epochs))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(tfrecord_non_overlap_parser, batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    # dataset = dataset.map(tfrecord_train_parser)
    # dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(batch_size)

    return dataset


def tfrecord_val_input_fn(filepath, batch_size, num_epochs):
    dataset = tf.data.TFRecordDataset(filepath)

    # Map the parser over dataset, and batch results by up to batch_size
    # dataset = dataset.shuffle(100000)
    # dataset = dataset.repeat(num_epochs)
    # dataset = dataset.map(tfrecord_train_parser)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(500, num_epochs))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(tfrecord_non_overlap_parser, batch_size, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    #dataset = dataset.prefetch(batch_size)

    return dataset


def tfrecord_test_input_fn(filepath, batch_size, num_epochs):
    dataset = tf.data.TFRecordDataset(filepath)

    # Map the parser over dataset, and batch results by up to batch_size
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(tfrecord_non_overlap_parser)
    # dataset = dataset.map(tfrecord_parser)
    dataset = dataset.batch(batch_size)

    return dataset
