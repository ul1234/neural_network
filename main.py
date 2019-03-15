#!/usr/bin/python
# -*- coding: utf-8 -*-

import mnist_loader
from network5 import *

class TestScenario(object):
    def __init__(self, net):
        self.net = net

    def set_epoch(self, epoches = 10):
        self.net.set_stop(EpochStop(epoches))

    def sigmoid_sigmoid_quadratic(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = Sigmoid, cost_func = Quadratic)
        self.net.set_train_func(Sgd(learning_rate))

    def sigmoid_sigmoid_crossentropy(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = Sigmoid, cost_func = CrossEntropy)
        self.net.set_train_func(Sgd(learning_rate))

    def sigmoid_softmax_loglikelihood(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = Sigmoid, last_layer_activation_func = Softmax, cost_func = Loglikelihood)
        self.net.set_train_func(Sgd(learning_rate))

    def ReLU_softmax_loglikelihood(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = ReLU, last_layer_activation_func = Softmax, cost_func = Loglikelihood)
        self.net.set_train_func(Sgd(learning_rate))

    def ReLU_softmax_loglikelihood_regularL2(self, learning_rate = 0.1, regular_lmda = 0.01):
        self.net.set_neuron(activation_func = ReLU, last_layer_activation_func = Softmax, cost_func = Loglikelihood)
        self.net.set_train_func(Sgd(learning_rate))
        self.net.set_regularization(RegularL2(regular_lmda))

    def tanh_softmax_loglikelihood(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = Tanh, last_layer_activation_func = Softmax, cost_func = Loglikelihood)
        self.net.set_train_func(Sgd(learning_rate))

    def sigmoid_sigmoid_crossentropy_momentum(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = Sigmoid, cost_func = CrossEntropy)
        self.net.set_train_func(MomentumSgd(learning_rate, coeffient = 0.5))

    def sigmoid_sigmoid_crossentropy_regularL2(self, learning_rate = 0.1, regular_lmda = 0.01):
        self.net.set_neuron(activation_func = Sigmoid, cost_func = CrossEntropy)
        self.net.set_train_func(Sgd(learning_rate))
        self.net.set_regularization(RegularL2(regular_lmda))

    def sigmoid_sigmoid_crossentropy_regularL1(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = Sigmoid, cost_func = CrossEntropy)
        self.net.set_train_func(Sgd(learning_rate))
        self.net.set_regularization(RegularL1(0.1))

    def sigmoid_sigmoid_crossentropy_dropout(self, learning_rate = 0.1):
        self.net.set_neuron(activation_func = Sigmoid, cost_func = CrossEntropy)
        self.net.set_train_func(Sgd(learning_rate))
        self.net.set_dropout(Dropout(0.5))

class TestNetwork(object):
    def __init__(self):
        self.training_data, self.validation_data, self.test_data = mnist_loader.load_data_wrapper()
        #self.training_data = self.training_data[:1000]   # small training data set
        self.test_data = Network.unpack_data(self.test_data)
        #self.net = Network([784, 30, 10])
        self.net = Network([784, 200, 10])
        self.test_scenario = TestScenario(self.net)

    def run(self, mini_batch_size = 10):
        self.net.train(self.training_data, mini_batch_size, test_data = self.test_data)

    def run_test(self, test, init_weight_func = WeightOpt()):
        self.net.init_weights(weight_func = init_weight_func)
        for func_args in test:
            func_args[0](*func_args[1:])
            self.net.reload_weights()
            print '\nTest for [%s%s]:\n' % (func_args[0].__name__, func_args[1:])
            self.run()

    def test_CrossEntropy_vs_Quadratic(self, learning_rate = 3.0):
        self.run_test([(self.test_scenario.sigmoid_sigmoid_quadratic, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_crossentropy, learning_rate)],
                       init_weight_func = WeightRandom(0.5))

    def test_softmax(self, learning_rate = 3.0):
        self.run_test([(self.test_scenario.sigmoid_softmax_loglikelihood, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_crossentropy, learning_rate)],
                       init_weight_func = WeightRandom(0.5))

    def test_ReLU(self, learning_rate = 0.1):
        self.run_test([(self.test_scenario.ReLU_softmax_loglikelihood, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_quadratic, learning_rate)])

    def test_tanh(self, learning_rate = 0.1):
        self.run_test([(self.test_scenario.tanh_softmax_loglikelihood, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_quadratic, learning_rate)])

    def test(self, learning_rate = 3.0):
        self.run_test([(self.test_scenario.sigmoid_sigmoid_quadratic, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_crossentropy, learning_rate),
                       (self.test_scenario.sigmoid_softmax_loglikelihood, learning_rate)])

    def test_best(self, learning_rate = 0.1, regular_lmda = 0.01):
        self.run_test([#(self.test_scenario.ReLU_softmax_loglikelihood, learning_rate),
                       (self.test_scenario.ReLU_softmax_loglikelihood_regularL2, learning_rate, regular_lmda),
                       #(self.test_scenario.sigmoid_sigmoid_crossentropy_regularL2, learning_rate, regular_lmda)
                       ])

    def test_momentum(self, learning_rate = 3.0):
        self.run_test([(self.test_scenario.sigmoid_sigmoid_quadratic_momentum, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_quadratic, learning_rate)])

    def test_regularization(self, learning_rate = 0.1):
        self.run_test([(self.test_scenario.sigmoid_sigmoid_crossentropy_regularL2, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_crossentropy_regularL1, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_crossentropy, learning_rate)],
                       init_weight_func = WeightRandom())  # should use 1000 training data for this test

    def test_dropout(self, learning_rate = 0.1):
        self.run_test([(self.test_scenario.sigmoid_sigmoid_crossentropy_dropout, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_crossentropy_regularL2, learning_rate),
                       (self.test_scenario.sigmoid_sigmoid_crossentropy, learning_rate)],
                       init_weight_func = WeightRandom())  # should use 1000 training data for this test


if __name__ == '__main__':
    test = TestNetwork()
    #test.test_scenario.set_epoch(100)
    #test.test_CrossEntropy_vs_Quadratic(20)
    #test.test_softmax()
    #test.test_ReLU()
    #test.test_tanh()
    #test.test()
    #test.test_momentum()
    #test.test_regularization(400, 0.5)
    test.test_best(learning_rate = 0.01, regular_lmda = 0.5)
    #test.test_dropout(0.5)


