from __future__ import division
import struct
import os
import xgboost as xgb
import numpy as np


# Code for reading MNIST files
def read_images(images_name):
    # Reading the image file
    f = open(images_name, "rb")
    ds_images = []
    # Header of the file
    mw_32bit = f.read(4)  # magic number
    n_numbers_32bit = f.read(4)  # number of images
    n_rows_32bit = f.read(4)  # number of rows
    n_columns_32bit = f.read(4)  # number of columns

    # Convert header to integers
    mw = struct.unpack('>i', mw_32bit)[0]
    n_numbers = struct.unpack('>i', n_numbers_32bit)[0]
    n_rows = struct.unpack('>i', n_rows_32bit)[0]
    n_columns = struct.unpack('>i', n_columns_32bit)[0]

    # Create set of images
    try:
        for i in range(n_numbers):
            image = []
            for r in range(n_rows):
                for l in range(n_columns):
                    byte = f.read(1)
                    pixel = struct.unpack('B', byte)[0]
                    image.append(pixel)
            ds_images.append(image)
    finally:
        f.close()
    return ds_images


def read_labels(labels_name):
    # Reading the label file
    f = open(labels_name, "rb")
    ds_labels = []
    # Header of the file
    mw_32bit = f.read(4)  # magic number
    n_numbers_32bit = f.read(4)  # number of labels

    # Convert header to integers
    mw = struct.unpack('>i', mw_32bit)[0]
    n_numbers = struct.unpack('>i', n_numbers_32bit)[0]

    # Create set of labels
    try:
        for i in range(n_numbers):
            byte = f.read(1)
            label = struct.unpack('B', byte)[0]
            ds_labels.append(label)

    finally:
        f.close()
    return ds_labels


def read_dataset(images_name, labels_name):
    # Reads both files and returns an array of tuples of
    # (flattened_image, label)
    images = read_images(images_name)
    labels = read_labels(labels_name)
    assert len(images) == len(labels)
    return (images, labels)


def create_datasets(sample_dir, nthread): # MNIST & XGBoost
    image_file = os.path.join(sample_dir, 'train-images-idx3-ubyte')
    label_file = os.path.join(sample_dir, 'train-labels-idx1-ubyte')
    training_images, training_labels = read_dataset(image_file, label_file)
    image_file = os.path.join(sample_dir, 't10k-images-idx3-ubyte')
    label_file = os.path.join(sample_dir, 't10k-labels-idx1-ubyte')
    testing_images, testing_labels = read_dataset(image_file, label_file)
    dtrain = xgb.DMatrix(
        np.asmatrix(training_images),
        label=training_labels,
        nthread=nthread
    )
    dtest = xgb.DMatrix(
        np.asmatrix(testing_images),
        label=testing_labels,
        nthread=nthread
    )
    data_dict = {
        'dtrain': dtrain,
        'dtest': dtest,
        'training_labels': training_labels,
        'testing_labels': testing_labels}
    return data_dict