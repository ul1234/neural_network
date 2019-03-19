#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from debug import Debug
from neuron import *
from train import *

########### weight initialize function ####################
class Weights(object):
    def __init__(self, w_shape, b_shape, activation, out_neuron_idx = -1):
        # w_shape:
        # FullConnectedLayer: (in_neurons, out_neurons)
        # ConvLayer: (out_depth, in_depth, filter_rows, filter_cols)
        # RNN Layer: [(in_neurons, state_neurons), (state_neurons, state_neurons), (state_neurons, out_neurons)]
        self.w_shape = w_shape
        self.b_shape = b_shape
        self.activation = activation
        self.set_method()
        # the index of out neuron in shape, used to calcualte the range of initial weights
        self.out_neuron_idx = out_neuron_idx

    def set_method(self, method = 'orthogonal', param = []):
        # method: 'opt', 'const', 'random'
        self.param = param
        self.init_weights = getattr(self, 'init_weights_{}'.format(method))
        self.init_biases = getattr(self, 'init_biases_{}'.format(method))

    def init(self):
        weights = [self.init_weights(w_shape) for w_shape in self.w_shape] \
            if isinstance(self.w_shape, list) else self.init_weights(self.w_shape)
        biases = [self.init_biases(b_shape) for b_shape in self.b_shape] \
            if isinstance(self.b_shape, list) else self.init_biases(self.b_shape)
        return weights, biases

    def init_weights_orthogonal(self, w_shape):
        # https://smerity.com/articles/2016/orthogonal_init.html
        activation = self.activation if isinstance(self.activation, list) else [self.activation]
        gain = np.sqrt(2) if ReLU in activation else 1.0
        assert len(w_shape) >= 2, 'invalid w_shape for orthogonal init'
        flat_shape = (w_shape[0], np.prod(w_shape[1:]))
        data = np.random.normal(0.0, 1.0, flat_shape)
        u, sigma, v = np.linalg.svd(data, full_matrices = False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(w_shape)
        return gain * q

    def init_biases_orthogonal(self, b_shape):
        return np.zeros(b_shape)

    def init_weights_opt(self, w_shape):
        return np.random.randn(*w_shape) / np.sqrt(np.prod(w_shape) / w_shape[self.out_neuron_idx])

    def init_biases_opt(self, b_shape):
        return np.zeros(b_shape)

    def init_weights_const(self, w_shape):
        w_const, b_const = self.param[0], self.param[1]
        return w_const * np.ones(w_shape)

    def init_biases_const(self, b_shape):
        w_const, b_const = self.param[0], self.param[1]
        return b_const * np.ones(b_shape)

    def init_weight_random(self, w_shape):
        err = self.param[0]
        return err + np.random.randn(w_shape)

    def init_biases_random(self, b_shape):
        err = self.param[0]
        return err + np.random.randn(b_shape)


########### layer ####################
class Layer(object):
    def __init__(self, weights_shape = [], biases_shape = [], activation = None, trainable = True, out_neuron_idx = -1):
        self.trainable = trainable
        self.activation = activation
        self.weights_shape = weights_shape
        self.biases_shape = biases_shape
        if trainable: self.weight_func = Weights(weights_shape, biases_shape, activation = activation, out_neuron_idx = out_neuron_idx)
        self.init()
        self.is_last_layer = False

    def set_cost_to_last_layer(self, cost):
        self.is_last_layer = True
        self.cost_func = cost

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

    def update_weights(self, training_size, optimizer, regularization = RegularNone()):
        if self.trainable:
            self.weights, self.biases = optimizer.update_weights(
                self.weights, self.biases, self.gradient_weights, self.gradient_biases, training_size, regularization)


class ConvLayer(Layer):
    def __init__(self, weights_shape, padding = 'valid'):
        self.out_depth, self.in_depth, self.filter_rows, self.filter_cols = weights_shape
        biases_shape = (self.out_depth, 1, 1)
        super().__init__(tuple(weights_shape), biases_shape, activation = ReLU, out_neuron_idx = 0)
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
        # data_in: shape (num_batches, in_depth, in_rows, in_cols)
        assert data_in.shape[1] == self.in_depth, 'invalid input depth'
        z = self._conv2d(data_in, self.weights) + self.biases
        data_out = self.activation.f(z)
        if in_back_propogation: self.data_in_for_backprop, self.data_out_for_backprop = data_in, data_out
        # data_out: shape (num_batches, out_depth, out_rows, out_cols)
        return data_out

    def back_propogation(self, delta):
        assert delta.shape == self.data_out_for_backprop.shape, 'invalid shape'
        # delta: 4D matrix, shape (num_batches, out_depth, out_rows, out_cols) -> (b, p, k, l)
        assert self.activation == ReLU, 'only ReLU is supported in convolutional layer now'
        num_batches = delta.shape[0]
        delta[self.data_out_for_backprop <= 0] = 0  # back propogation for ReLU
        self.gradient_biases = delta.sum(axis = (0,2,3)).reshape(self.biases.shape) / num_batches
        self.gradient_weights = self._back_conv2d(self.data_in_for_backprop, delta) / num_batches
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
        super().__init__(tuple(weights_shape), biases_shape, activation = activation)
        self.dropout = dropout

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
        self.gradient_weights = np.dot(self.data_in_for_backprop.T, delta) / num_batches
        self.gradient_biases = np.dot(np.ones((1, num_batches)), delta) / num_batches
        data_out = data_out.reshape(self.data_in_shape_for_backprop)
        return data_out

# 1) seems using RNN, the state_neurons should be larger. If not, it's hard to train a good performance
class RecurrentLayer(Layer):
    def __init__(self, in_neurons, state_neurons, out_neurons, only_use_last_output_t = True, activation = [Tanh, Sigmoid]):
        self.in_neurons, self.state_neurons, self.out_neurons = in_neurons, state_neurons, out_neurons
        weights_shape = [(in_neurons, state_neurons), (state_neurons, state_neurons), (state_neurons, out_neurons)]
        biases_shape = [(1, state_neurons), (1, out_neurons)]
        super().__init__(weights_shape, biases_shape, activation = activation)
        self.state = np.zeros((1, state_neurons))
        self.only_use_last_output_t = only_use_last_output_t

    def _feedforward(self, data_in, in_back_propogation = False):
        # data_in: shape (num_batches, in_neurons)
        in_weights, state_weights, out_weights = self.weights
        state_biases, out_biases = self.biases
        state_activation, out_activation = self.activation
        new_state = np.dot(data_in, in_weights) + np.dot(self.state, state_weights) + state_biases
        # state: shape (num_batches, state_neurons)
        self.state = state_activation.f(new_state)
        data_out = np.dot(self.state, out_weights) + out_biases
        data_out = out_activation.f(data_out)
        # data_out: shape (num_batches, out_neurons)
        return data_out, self.state

    def feedforward(self, data_in, in_back_propogation = False):
        # data_in: shape (num_batches, num_recur, in_neurons)
        num_batches, num_recur, in_neurons = data_in.shape
        assert in_neurons == self.in_neurons, 'invalid in_neurons'
        # reset state to zeros
        self.state = np.zeros((num_batches, self.state_neurons))
        data_in = np.transpose(data_in, axes = [1, 0, 2])
        data_out = np.zeros((num_recur, num_batches, self.out_neurons))
        state = np.zeros((num_recur+1, num_batches, self.state_neurons))
        for t in range(num_recur):
            data_out[t], state[t+1] = self._feedforward(data_in[t], in_back_propogation = in_back_propogation)
        if self.only_use_last_output_t: data_out = data_out[-1] # data_out: shape (num_batches, out_neurons)
        if in_back_propogation:
            self.data_in_for_backprop = data_in
            self.data_out_for_backprop = data_out
            self.state_for_backprop = state
            self.num_recur_for_backprop = num_recur
        # data_out: shape (num_batches, num_recur, out_neurons)
        if not self.only_use_last_output_t: data_out = np.transpose(data_out, axes = [1, 0, 2])
        return data_out

    def _back_propogation(self, delta, delta_state, recur_idx):
        # if only use the last recurrent output, we can ignore delta (set to 0) for recurrent index other than the last recurrent index
        ignore_delta = True if self.only_use_last_output_t and recur_idx < (self.num_recur_for_backprop - 1) else False
        # delta: shape (num_batches, out_neurons)
        num_batches, out_neurons = delta.shape
        assert out_neurons == self.out_neurons, 'invalid out_neurons'
        in_weights, state_weights, out_weights = self.weights
        state_biases, out_biases = self.biases
        state_activation, out_activation = self.activation
        if ignore_delta:
            gradient_out_weights = np.zeros_like(out_weights)
            gradient_out_biases = np.zeros_like(out_biases)
        else:
            if self.is_last_layer:
                # if the last layer, delta is actually the labelled data, i.e. y
                delta = self.cost_func.delta(self.data_out_for_backprop, delta)
            else:
                delta *= out_activation.derivative_a(self.data_out_for_backprop)
            gradient_out_weights = np.dot(self.state_for_backprop[recur_idx+1].T, delta) / num_batches
            gradient_out_biases = np.dot(np.ones((1, num_batches)), delta) / num_batches
            # delta_state should add the part back propogated from delta
            # delta_state: shape (num_batches, state_neurons)
            delta_state += np.dot(delta, out_weights.T)
        delta_state *= state_activation.derivative_a(self.state_for_backprop[recur_idx+1])
        gradient_in_weights = np.dot(self.data_in_for_backprop[recur_idx].T, delta_state) / num_batches
        gradient_state_weights = np.dot(self.state_for_backprop[recur_idx].T, delta_state) / num_batches
        gradient_state_biases = np.dot(np.ones((1, num_batches)), delta_state) / num_batches
        # data_out: shape (num_batches, in_neurons)
        data_out = np.dot(delta_state, in_weights.T)
        # delta_previous_state: shape (num_batches, state_neurons)
        delta_previous_state = np.dot(delta_state, state_weights.T)
        gradient_weights_t = [gradient_in_weights, gradient_state_weights, gradient_out_weights]
        gradient_biases_t = [gradient_state_biases, gradient_out_biases]
        return data_out, delta_previous_state, gradient_weights_t, gradient_biases_t

    def back_propogation(self, delta):
        assert self.data_out_for_backprop.shape == delta.shape, 'invalid shape'
        if self.only_use_last_output_t:
            # delta: shape (num_batches, out_neurons)
            num_batches, out_neurons = delta.shape
        else:
            # delta: shape (num_batches, num_recur, out_neurons)
            num_batches, num_recur, out_neurons = delta.shape
            # delta: shape (num_recur, num_batches, out_neurons)
            delta = np.transpose(delta, axes = [1, 0, 2])
        delta_state = np.zeros((num_batches, self.state_neurons))
        # data_out: shape (num_recur, num_batches, in_neurons)
        data_out = np.zeros((self.num_recur_for_backprop, num_batches, self.in_neurons))
        self.gradient_weights = [np.zeros(w.shape) for w in self.weights]
        self.gradient_biases = [np.zeros(b.shape) for b in self.biases]
        for t in range(self.num_recur_for_backprop)[::-1]:
            in_delta = delta if self.only_use_last_output_t else delta[t]
            data_out[t], delta_state, gradient_weights_t, gradient_biases_t = self._back_propogation(in_delta, delta_state, t)
            self.gradient_weights = [w + w_t for w, w_t in zip(self.gradient_weights, gradient_weights_t)]
            self.gradient_biases = [b + b_t for b, b_t in zip(self.gradient_biases, gradient_biases_t)]
        self.gradient_clipping()
        # data_out: shape (num_batches, num_recur, out_neurons)
        data_out = np.transpose(data_out, axes = [1, 0, 2])
        return data_out

    def gradient_clipping(self, clipping_value = 10):
        gradient_w_need_clipping = any([np.any(np.abs(gradient_w) > clipping_value) for gradient_w in self.gradient_weights])
        gradient_b_need_clipping = any([np.any(np.abs(gradient_b) > clipping_value) for gradient_b in self.gradient_biases])
        if gradient_w_need_clipping or gradient_b_need_clipping:
            print('clipping needed. w: {}, b: {}'.format(gradient_w_need_clipping, gradient_b_need_clipping))
        #for gradient_weights in self.gradient_weights:
        #    gradient_weights[gradient_weights>clipping_value] = clipping_value
        #    gradient_weights[gradient_weights<-clipping_value] = -clipping_value
        #for gradient_biases in self.gradient_biases:
        #    gradient_biases[gradient_biases>clipping_value] = clipping_value
        #    gradient_biases[gradient_biases<-clipping_value] = -clipping_value

