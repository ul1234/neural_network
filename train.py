#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time, json
from debug import Debug
from neuron import *


########### regularization ####################
class Regularization(object):
    def __init__(self, lmda = 0.1):
        self.lmda = lmda

    def cost(self, weights, data_size):
        raise

    def update_weights(self, weights, learning_rate, total_training_size):
        raise

class RegularNone(Regularization):
    def cost(self, weights, data_size):
        return 0

    def update_weights(self, weights, learning_rate, total_training_size):
        return weights

class RegularL2(Regularization):
    def cost(self, weights, data_size):
        return self.lmda / 2 / data_size * sum([np.square(w).sum() for w in weights])

    def update_weights(self, weights, learning_rate, total_training_size):
        return (1 - self.lmda * learning_rate / total_training_size) * weights

class RegularL1(Regularization):
    def cost(self, weights, data_size):
        return self.lmda / data_size * sum([np.abs(w).sum() for w in weights])

    def update_weights(self, weights, learning_rate, total_training_size):
        return weights - self.lmda * learning_rate / total_training_size * np.sign(weights)


class Dropout(object):
    def __init__(self, drop_probability = 0.5):
        self.drop_probability = drop_probability
        self.weight_factor = 1

    def drop(self, data_out):
        # data_out: shape (num_batches, out_neurons)
        num_batches, out_neurons = data_out.shape
        num_drop = int(out_neurons * self.drop_probability)
        self.weight_factor = 1 - num_drop / out_neurons
        index = range(out_neurons)
        for batch in range(num_batches):
            np.random.shuffle(index)
            drop_index = index[:num_drop]
            data_out[batch, drop_index] = 0

    def compensate_weight(self, weight):
        return weight * self.weight_factor


########### train method ####################
class Optimizer(object):
    pass

class MomentumSgd(Optimizer):
    def __init__(self, learning_rate = 0.1, coeffient = 0.5):
        self.learning_rate = learning_rate
        self.coeffient = coeffient

    def init(self, sizes):
        num_layers = len(sizes)
        self.weights_velocity = [np.zeros((sizes[layer], sizes[layer-1])) for layer in range(1, num_layers)]
        self.biases_velocity = [np.zeros((sizes[layer], 1)) for layer in range(1, num_layers)]

    def update_weights(self, weights, biases, delta_w, delta_b, data_size, total_training_size, regularization = RegularNone()):
        self.weights_velocity = [self.coeffient * wv - self.learning_rate / data_size * dwv for wv, dwv in zip(self.weights_velocity, delta_w)]
        self.biases_velocity = [self.coeffient * bv - self.learning_rate / data_size * dbv for bv, dbv in zip(self.biases_velocity, delta_b)]
        weights = [regularization.update_weights(w, self.learning_rate, total_training_size) + wv for w, wv in zip(weights, self.weights_velocity)]
        biases = [b + bv for b, bv in zip(biases, self.biases_velocity)]
        return weights, biases

class Sgd(Optimizer):
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def update_weights(self, weights, biases, delta_w, delta_b, data_size, total_training_size, regularization = RegularNone()):
        weights = regularization.update_weights(weights, self.learning_rate, total_training_size) - self.learning_rate / data_size * delta_w
        biases = biases - self.learning_rate / data_size * delta_b
        return weights, biases

########### early stopping ####################
class EarlyStop(object):
    def __init__(self, check_epoches = 10):
        # for some epoches that the test accuracy do not get a new larger value
        self.check_epoches = check_epoches
        self.accuracy_history = []

    def stop(self, test_accuracy = None):
        self.accuracy_history.append(test_accuracy)
        if len(self.accuracy_history) >= self.check_epoches:
            last_accuracy = self.accuracy_history[-self.check_epoches:]
            return np.argmax(last_accuracy) == 0
        return False

class EpochStop(object):
    def __init__(self, stop_epoches = 30):
        self.stop_epoches = stop_epoches
        self.run_epoches = 0

    def stop(self, test_accuracy = None):
        self.run_epoches += 1
        if (self.run_epoches >= self.stop_epoches):
            self.run_epoches = 0
            return True
        return False

