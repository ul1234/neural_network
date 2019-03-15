#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from neuron import *
from train import *

########### weight initialize function ####################
class Weights(object):
    def __init__(self, w_shape, b_shape):
        # w_shape: (in_neurons, out_neurons) or (out_depth, in_depth, filter_rows, filter_cols)
        self.w_shape = w_shape
        self.b_shape = b_shape
        self.set_method()

    def set_method(self, method = 'opt', param = []):
        self.param = param
        self.init = getattr(self, 'init_{}'.format(method))

    def init_opt(self):
        in_neurons = np.prod(self.w_shape) / np.prod(self.b_shape)
        weights = np.random.randn(*self.w_shape) / np.sqrt(in_neurons)
        #biases = np.random.randn(self.w_shape[1], 1)
        biases = np.zeros(self.b_shape)
        return weights, biases

    def init_const(self):
        w_const, b_const = self.param[0], self.param[1]
        weights = w_const * np.ones(self.w_shape)
        biases = b_const * np.ones(self.b_shape)
        return weights, biases

    def init_random(self):
        err = self.param[0]
        weights = err + np.random.randn(self.w_shape)
        biases = err + np.random.randn(self.b_shape)
        return weights, biases


########### layer ####################
class Layer(object):
    def __init__(self, weights_shape = [], biases_shape = [], activation = None, trainable = True):
        self.trainable = trainable
        self.activation = activation
        self.weights_shape = weights_shape
        self.biases_shape = biases_shape
        if trainable: self.weight_func = Weights(weights_shape, biases_shape)
        self.init()

    def init(self):
        if self.trainable:
            self.weights, self.biases = self.weight_func.init()

    def set_weight_init(self, method = 'opt', param = []):
        self.weight_func.set_method(method, param)
        self.init()

    def feedforward(self, data_in, in_back_propogation = False):
        raise

    def back_propogation(self, delta):
        raise

    def update_weights(self, mini_batch_data_size, training_size, optimizer, regularization = RegularNone()):
        if self.trainable:
            self.weights, self.biases = optimizer.update_weights(
                self.weights, self.biases, self.delta_w, self.delta_b, mini_batch_data_size, training_size, regularization)

    def get_weights(self):
        if self.trainable:
            pass
            

class ConvLayer(Layer):
    def __init__(self, weights_shape, padding = 'valid'):
        self.out_depth, self.in_depth, self.filter_rows, self.filter_cols = weights_shape
        biases_shape = (self.out_depth, 1, 1)
        super().__init__(weights_shape, biases_shape, activation = ReLU)
        assert padding == 'valid', 'only valid is supported now'

    def _conv2d(self, data_in, filter2d):
        # refer: https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
        # data_in: 4D matrix, shape (num_batches, in_depth, in_rows, in_cols) -> (b, q, m, n)
        # filter2d: 4D matrix, shape (out_depth, in_depth, filter_rows, filter_cols) -> (p, q, i, j)
        # data_out: 4D matrix, shape (num_batches, out_depth, out_rows, out_cols) -> (b, p, m-i+1, n-j+1) -> (b, p, k, l)
        num_batches, in_data_depth, in_rows, in_cols = data_in.shape
        out_depth, in_depth, filter_rows, filter_cols = filter2d.shape
        assert in_data_depth == in_depth, 'invalid in_depth'
        out_rows, out_cols = in_rows - filter_rows + 1, in_cols - filter_cols + 1
        view_shape = (num_batches, out_rows, out_cols, in_depth, filter_rows, filter_cols)  # (b, k, l, q, i, j)
        batch_stride, in_depth_stride, in_rows_stride, in_cols_stride = data_in.strides
        view_strides = (batch_stride, in_rows_stride, in_cols_stride, in_depth_stride, in_rows_stride, in_cols_stride)
        view_matrix = np.lib.stride_tricks.as_strided(data_in, view_shape, view_strides, writeable = False)
        return np.einsum('pqij,bklqij->bpkl', filter2d, view_matrix)

    def _back_conv2d(self, data_in, filter2d):
        # data_in: 4D matrix, shape (num_batches, in_depth, in_rows, in_cols) -> (b, q, m, n)
        # filter2d: 4D matrix, shape (num_batches, out_depth, out_rows, out_cols) -> (b, p, k, l)
        # data_out: 4D matrix shape (out_depth, in_depth, filter_rows, filter_cols) -> (p, q, m-k+1, n-l+1) -> (p, q, i, j)
        num_in_batches, in_depth, in_rows, in_cols = data_in.shape
        num_batches, out_depth, out_rows, out_cols = filter2d.shape
        assert num_batches == num_in_batches, 'invalid num_batches'
        filter_rows, filter_cols = in_rows - out_rows + 1, in_cols - out_cols + 1
        view_shape = (num_batches, filter_rows, filter_cols, in_depth, out_rows, out_cols)  # (b, i, j, q, k, l)
        batch_stride, in_depth_stride, in_rows_stride, in_cols_stride = data_in.strides
        view_strides = (batch_stride, in_rows_stride, in_cols_stride, in_depth_stride, in_rows_stride, in_cols_stride)
        view_matrix = np.lib.stride_tricks.as_strided(data_in, view_shape, view_strides, writeable = False)
        return np.einsum('bpkl,bijqkl->pqij', filter2d, view_matrix)

    def feedforward(self, data_in, in_back_propogation = False):
        assert data_in.shape[1] == self.in_depth, 'invalid input depth'
        z = self._conv2d(data_in, self.weights) + self.biases
        data_out = self.activation.f(z)
        if in_back_propogation: self.data_in_for_backprop, self.data_out_for_backprop = data_in, data_out
        return data_out

    def back_propogation(self, delta):
        assert delta.shape == self.data_out_for_backprop.shape, 'invalid shape'
        # delta: 4D matrix, shape (num_batches, out_depth, out_rows, out_cols) -> (b, p, k, l)
        assert self.activation == ReLU, 'only ReLU is supported in convolutional layer now'
        delta[self.data_out_for_backprop <= 0] = 0  # back propogation for ReLU
        self.delta_b = delta.sum(axis = (0,2,3)).reshape(self.biases.shape)
        self.delta_w = self._back_conv2d(self.data_in_for_backprop, delta)
        pad_rows, pad_cols = self.filter_rows - 1, self.filter_cols - 1
        # delta_pad: (num_batches, out_depth, in_rows + filter_rows -1, in_cols + filter_cols - 1)
        delta_pad = np.pad(delta, ((0, 0), (0, 0), (pad_rows, pad_rows), (pad_cols, pad_cols)), 'constant', constant_values = 0)
        # rotate the weight filters by 180 degrees, for convolution, shape (out_depth, in_depth, filter_rows, filter_cols) -> (p, q, i, j)
        back_weights = np.rot90(self.weights, 2, axes = (2,3))
        # turn the shape to (q, p, i, j), in order to reuse conv2d() function
        back_weights = np.transpose(back_weights, axes = [1, 0, 2, 3])
        # delta_pad: shape (num_batches, out_depth, in_rows + filter_rows -1, in_cols + filter_cols - 1) -> (b, p, m+i-1, n+j-1)
        # back_weights: shape (in_depth, out_depth, filter_rows, filter_cols) -> (q, p, i, j)
        # data_out: shape (num_batches, in_depth, in_rows, in_cols) -> (b, q, m, n)
        data_out = self._conv2d(delta_pad, back_weights)
        return data_out

class PoolingLayer(Layer):
    def __init__(self, strides = 2, method = 'max'):
        super().__init__(trainable = False)
        self.strides = strides
        assert method == 'max', 'only max pooling is supported now'

    def feedforward(self, data_in, in_back_propogation = False):
        # data_in: 4D matrix, shape (num_batches, in_depth, in_rows, in_cols) -> (b, q, m, n)
        num_batches, in_depth, in_rows, in_cols = data_in.shape
        view_shape = (num_batches, in_depth, int(in_rows/self.strides), int(in_cols/self.strides), self.strides, self.strides)
        batch_stride, in_depth_stride, in_rows_stride, in_cols_stride = data_in.strides
        view_strides = (batch_stride, in_depth_stride, in_rows_stride*self.strides, in_cols_stride*self.strides, in_rows_stride, in_cols_stride)
        # view_matrix: 6D matrix, shape (b, q, m/stride, n/stride, stride, stride)
        view_matrix = np.lib.stride_tricks.as_strided(data_in, view_shape, view_strides, writeable = False)
        # dataout: shape (num_batches, in_depth, out_rows, out_cols) -> (b, q, m/stride, n/stride) -> (b, p, k, l)
        data_out =  view_matrix.max(axis = (4,5))
        if in_back_propogation:
            # save argmax index for back propogation
            view_matrix = view_matrix.reshape(*data_out.shape, -1)
            self.data_out_index_for_backprop = np.argmax(view_matrix, axis = -1)
            self.data_out_for_backprop = data_out
        return data_out

    def back_propogation(self, delta):
        assert delta.shape == self.data_out_for_backprop.shape, 'invalid shape'
        # delta: 4D matrix, shape (num_batches, in_depth, out_rows, out_cols) -> (b, p, k, l)
        num_batches, in_depth, out_rows, out_cols = delta.shape
        data_shape = (num_batches, in_depth, out_rows, out_cols, self.strides*self.strides)
        data_out = np.zeros(data_shape).reshape(-1, data_shape[-1])
        data_out[range(data_out.shape[0]), np.ravel(self.data_out_index_for_backprop)] = np.ravel(delta)
        data_out_shape = (num_batches, in_depth, out_rows*self.strides, out_cols*self.strides)
        # data_out: 4D matrix, shape (num_batches, in_depth, in_rows, in_cols) -> (b, q, k*strides, l*strides) -> (b, q, m, n)
        data_out = data_out.reshape(data_out_shape)
        return data_out

class FullConnectedLayer(Layer):
    def __init__(self, weights_shape, activation = Sigmoid, dropout = None):
        self.in_neurons, self.out_neurons = weights_shape
        biases_shape = (1, self.out_neurons)
        super().__init__(weights_shape, biases_shape, activation = activation)
        self.dropout = dropout
        self.is_last_layer = False

    def set_cost_to_last_layer(self, cost):
        self.is_last_layer = True
        self.cost_func = cost

    def feedforward(self, data_in, in_back_propogation = False):
        # data_in: shape (num_batches, in_neurons) or (num_batches, in_depth, in_rows, in_cols)
        self.data_in_shape_for_backprop = data_in.shape
        num_batches, in_neurons = data_in.shape[0], np.prod(data_in.shape[1:])
        assert in_neurons == self.in_neurons, 'invalid data_in size'
        weights = self.dropout.compensate_weight(self.weights) if self.dropout and not in_back_propogation else self.weights
        # data_in: shape (num_batches, in_neurons)
        data_in = data_in.reshape(-1, in_neurons)
        z = np.dot(data_in, weights) + self.biases
        # data_out: shape (num_batches, out_neurons)
        data_out = self.activation.f(z)
        if in_back_propogation:
            if self.dropout: self.dropout.drop(data_out)
            self.data_in_for_backprop, self.data_out_for_backprop = data_in, data_out
        return data_out

    def back_propogation(self, delta):
        # delta: shape (num_batches, out_neurons)
        num_batches, out_neurons = delta.shape
        assert delta.shape == self.data_out_for_backprop.shape, 'invalid shape'
        if self.is_last_layer:
            # if the last layer, delta is actually the labelled data, i.e. y
            delta = self.cost_func.delta(self.data_out_for_backprop, delta)
        else:
            delta *= self.activation.derivative_a(self.data_out_for_backprop)
        # data_out: shape (num_batches, in_neurons)
        data_out = np.dot(delta, self.weights.T)
        self.delta_w = np.dot(self.data_in_for_backprop.T, delta)
        self.delta_b = np.dot(np.ones((1, num_batches)), delta)
        data_out = data_out.reshape(self.data_in_shape_for_backprop)
        return data_out
