#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


########### activation function ####################
class Activation(object): pass

class Sigmoid(Activation):
    @staticmethod
    def f(z):
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def derivative(z):
        a = f.__func__(z)
        return a * (1 - a)

    @staticmethod
    def derivative_a(a):
        return a * (1 - a)

class Tanh(Activation):
    @staticmethod
    def f(z):
        t1 = np.exp(z)
        t2 = np.exp(-z)
        return (t1 - t2) / (t1 + t2)

    @staticmethod
    def derivative_a(a):
        return 1 - a * a

# Seems when using ReLU:
# 1) small standardized initial weights and zero initial biases should be used
# 2) also smaller learning rate
# 3) the performance greatly depends on the initial weights, seems some local minimum exist???
class ReLU(Activation):
    @staticmethod
    def f(z):
        temp = z.copy()
        temp[temp < 0] = 0
        return temp

    @staticmethod
    def derivative_a(a):
        temp = a.copy()
        temp[temp <= 0] = 0
        temp[temp > 0] = 1
        return temp

# only used in the last layer, so no need for the derivative_a function
class Softmax(Activation):
    @staticmethod
    def f(z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis = 0)


########### cost function ####################
class Cost(object):
    def __init__(self, activation = None):
        self.activation = activation

    def cost(self, a, y):
        raise

    def delta(self, a, y):
        raise

class Loglikelihood(Cost):
    def cost(self, a, y):
        return -(y * np.log(a)).sum(axis = 0).mean()

    def delta(self, a, y):
        assert self.activation == Softmax, 'only Softmax is supported with Loglikelihood now'
        return a - y

class Quadratic(Cost):
    def cost(self, a, y):
        return np.square(a - y).mean()

    def derivative(self, a, y):
        return (a - y)

    def delta(self, a, y):
        return (a - y) * self.activation.derivative_a(a)

class CrossEntropy(Cost):
    def cost(self, a, y):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a)).mean()

    def derivative(self, a, y):
        return (a - y) / ( a * (1 - a))

    def delta(self, a, y):
        assert self.activation == Sigmoid, 'only Sigmoid is supported with CrossEntropy now'
        return (a - y)
