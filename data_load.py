#!/usr/bin/python
# -*- coding: utf-8 -*-

import gzip, os, pickle
import numpy as np

class DataLoad(object):
    def __init__(self, path = 'data'):
        self.path = path

    def _load_mnist(self):
        # training_data, validation_data, test_data: 50000, 10000, 10000 images
        # type is (input_data, label_data)
        # input_data: numpy ndarray, num_entries * 28 * 28 (pixels)
        # label_data: numpy ndarray, num_entries * 1 (digital value 0~9)
        f = gzip.open(os.path.join(self.path, 'mnist.pkl.gz'), 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding = 'bytes')
        f.close()
        return (training_data, validation_data, test_data)

    def load_mnist(self, shape = (-1, -1)):
        # training_data, validation_data, test_data: 50000, 10000, 10000 images
        # type is [(input_data1, label_data1), (input_data2, label_data2), ...]
        # input_data1: numpy ndarray, shape (28*28,) (pixels)
        # label_data1: numpy ndarray, shape(10,)
        label_matrix = np.eye(10)
        x_shape, y_shape = shape
        return tuple([[(x.reshape(x_shape), label_matrix[y].reshape(y_shape)) for x, y in zip(*data)] for data in self._load_mnist()])

