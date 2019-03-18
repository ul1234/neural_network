#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pprint
from debug import *
from network import *


class TestNetwork(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
        
    def xx_test_FullConnected(self):
        fc_layer = FullConnectedLayer([3, 2])
        net = Network([fc_layer],
                       cost = CrossEntropy(Sigmoid),
                       optimizer = Sgd(0.1))
        input_data = np.random.rand(4,3)
        label_data = np.array([[1,0], [1,0], [0,1], [0,1]])
        self._gradient_check(net, input_data, label_data)
        
    def xx_test_Conv(self):
        conv_layer = ConvLayer([2,2,3,3])
        #pooling_layer = PoolingLayer()
        fc_layer = FullConnectedLayer([32, 10])
        net = Network([conv_layer, fc_layer],
                       cost = CrossEntropy(Sigmoid),
                       optimizer = Sgd(0.1))
        input_data = np.random.rand(1, 2, 6, 6)
        label_data = np.array([1,0,0,0,0,0,0,0,0,0])[np.newaxis, :]
        self._gradient_check(net, input_data, label_data)
        
    def test_RNN(self):
        rnn_layer1 = RecurrentLayer(5, 5, 10)
        fc_layer1 = FullConnectedLayer([10, 10])
        net = Network([rnn_layer1, fc_layer1],
                      # [rnn_layer1],
                       cost = CrossEntropy(Sigmoid),
                       optimizer = Sgd(0.1))
        input_data = np.random.rand(2, 5, 5)
        label_data = np.array([[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0]])
        #label_data = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]])
        self._gradient_check(net, input_data, label_data)
        
    def _gradient_check(self, net, input_data, label_data):
        net.feedforward(input_data, in_back_propogation = True)
        net.back_propogation(label_data)
        self._calc_gradient(net, net.pack_data(input_data, label_data))
        for layer in net.layers:
            if not layer.trainable: continue
            Debug.print_('layer:',  layer.__class__.__name__, 'gradient_weights:', layer.gradient_weights, 'gradient_weights_est:', layer.gradient_weights_est)
            gradient_weights = layer.gradient_weights if isinstance(layer.gradient_weights, list) else [layer.gradient_weights]
            gradient_weights_est = layer.gradient_weights_est if isinstance(layer.gradient_weights_est, list) else [layer.gradient_weights_est]
            for gradient_w, gradient_w_est in zip(gradient_weights, gradient_weights_est):
                np.testing.assert_allclose(gradient_w, gradient_w_est, rtol = 0, atol = 1e-4)
            Debug.print_('layer:',  layer.__class__.__name__, 'gradient_biases:', layer.gradient_biases, 'gradient_biases_est:', layer.gradient_biases_est)
            gradient_biases = layer.gradient_biases if isinstance(layer.gradient_biases, list) else [layer.gradient_biases]
            gradient_biases_est = layer.gradient_biases_est if isinstance(layer.gradient_biases_est, list) else [layer.gradient_biases_est]
            for gradient_b, gradient_b_est in zip(gradient_biases, gradient_biases_est):
                np.testing.assert_allclose(gradient_b, gradient_b_est, rtol = 0, atol = 1e-4)
        
    def _calc_gradient(self, net, pack_data):
        delta_weights = delta_biases = 0.0001
        for layer in net.layers:
            if not layer.trainable: continue
            weights_list = layer.weights if isinstance(layer.weights, list) else [layer.weights]
            gradient_weights_est = [np.zeros_like(weights) for weights in weights_list]
            for weights_list_index, weights in enumerate(weights_list):
                for weights_index in range(weights.size):
                    np.ravel(gradient_weights_est[weights_list_index])[weights_index] = self._gradient_weights_est(pack_data, net, weights, weights_index, delta_weights)
            layer.gradient_weights_est = gradient_weights_est if isinstance(layer.weights, list) else gradient_weights_est[0]
            # calc biases gradient
            biases_list = layer.biases if isinstance(layer.biases, list) else [layer.biases]
            gradient_biases_est = [np.zeros_like(biases) for biases in biases_list]
            for biases_list_index, biases in enumerate(biases_list):
                for biases_index in range(biases.size):
                    np.ravel(gradient_biases_est[biases_list_index])[biases_index] = self._gradient_biases_est(pack_data, net, biases, biases_index, delta_biases)
            layer.gradient_biases_est = gradient_biases_est if isinstance(layer.biases, list) else gradient_biases_est[0]
                    
    def _gradient_weights_est(self, pack_data, net, weights, weights_index, delta_weights):
        np.ravel(weights)[weights_index] += delta_weights
        _, cost1 = net.accuracy(pack_data)
        np.ravel(weights)[weights_index] -= 2*delta_weights
        _, cost2 = net.accuracy(pack_data)
        gradient_weights_est = (cost1 - cost2) / (2*delta_weights)  
        np.ravel(weights)[weights_index] += delta_weights
        return gradient_weights_est
                    
    def _gradient_biases_est(self, pack_data, net, biases, biases_index, delta_biases):
        np.ravel(biases)[biases_index] += delta_biases
        _, cost1 = net.accuracy(pack_data)
        np.ravel(biases)[biases_index] -= 2*delta_biases
        _, cost2 = net.accuracy(pack_data)
        gradient_biases_est = (cost1 - cost2) / (2*delta_biases)  
        np.ravel(biases)[biases_index] += delta_biases
        return gradient_biases_est


if __name__ == '__main__':
    Debug.ENABLE = True
    unittest.main()