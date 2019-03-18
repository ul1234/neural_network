#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pprint
from debug import *
from network import *

# This file do not work now after the network codes are refactored
# todo later to make it work

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
        input_data = np.array([0.1, 0.2, 0.3])[np.newaxis, :]
        label_data = np.array([1,0])[np.newaxis, :]
        self._gradient_check(net, input_data, label_data)
        
    def xx_test_Conv(self):
        conv_layer = ConvLayer([2,2,3,3])
        #pooling_layer = PoolingLayer()
        #self.fc_layer = FcLayer([8, 30, 1])
        fc_layer = FullConnectedLayer([32, 10])
        net = Network([conv_layer, fc_layer],
                       cost = CrossEntropy(Sigmoid),
                       optimizer = Sgd(0.1))
        input_data = np.array([0.74341247,  0.47463755,  0.00992929,  0.95730784,  0.20542349,
                                0.24606582,  0.5627104 ,  0.18438329,  0.89370057,  0.73840308,
                                0.96136674,  0.19538822,  0.1619067 ,  0.02462808,  0.85983933,
                                0.92236065,  0.87674389,  0.68733282,  0.4138197 ,  0.41656749,
                                0.38043692,  0.78814061,  0.30552122,  0.44576086,  0.79040761,
                                0.78019093,  0.95638804,  0.2221817 ,  0.18427876,  0.53748266,
                                0.23379542,  0.27326781,  0.14063543,  0.24078563,  0.54046106,
                                0.09593265]).reshape(1, 6, 6);
        input_data = np.concatenate((input_data, 1-input_data)).reshape(1, 2, 6, 6)
        label_data = np.array([1,0,0,0,0,0,0,0,0,0])[np.newaxis, :]
        self._gradient_check(net, input_data, label_data)
        
    def test_RNN(self):
        rnn_layer1 = RecurrentLayer(2, 3, 2)
        net = Network(#[rnn_layer1, fc_layer1],
                       [rnn_layer1],
                       cost = CrossEntropy(Sigmoid),
                       optimizer = Sgd(0.1))
        input_data = np.array([0.1, 0.2])[np.newaxis, np.newaxis, :]
        label_data = np.array([1, 0])[np.newaxis, :]
        self._gradient_check(net, input_data, label_data)
        
    def _gradient_check(self, net, input_data, label_data):
        net.feedforward(input_data, in_back_propogation = True)
        net.back_propogation(label_data)
        self._calc_gradient(net, net.pack_data(input_data, label_data))
        for layer in net.layers:
            if not layer.trainable: continue
            Debug.print_('layer:',  layer.__class__.__name__, 'gradient_weights:', layer.gradient_weights, 'gradient_weights_est:', layer.gradient_weights_est)
            np.testing.assert_allclose(layer.gradient_weights, layer.gradient_weights_est, rtol = 1e-3)
            Debug.print_('layer:',  layer.__class__.__name__, 'gradient_biases:', layer.gradient_biases, 'gradient_biases_est:', layer.gradient_biases_est)
            np.testing.assert_allclose(layer.gradient_biases, layer.gradient_biases_est, rtol = 1e-3)
        
    def _calc_gradient(self, net, pack_data):
        delta_weights = delta_biases = 0.0001
        for layer in net.layers:
            if not layer.trainable: continue
            weights_list = layer.weights if isinstance(layer.weights, list) else [layer.weights]
            gradient_weights_est = [np.zeros_like(weights) for weights in weights_list]
            for weights_list_index, weights in enumerate(weights_list):
                for weights_index in range(weights.size):
                    if (Debug.count() == 19):
                        a = 0
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

        
    
class TestBackProp(object):
    def __init__(self, layers, layer_input_a):
        self.layers = [layers] if not isinstance(layers, list) else layers
        self.layer_input_a = layer_input_a
        self.cost_func = layers[-1].cost_func if hasattr(layers[-1], 'cost_func') else None

    def get_layer_weights(self, layer):
        if isinstance(layer.weights, list):
            return layer.weights[:], layer.biases[:]
        else:
            return layer.weights.copy(), layer.biases.copy()

    def get_layer_deriv_weights(self, layer):
        if isinstance(layer.delta_w, list):
            return layer.delta_w, layer.delta_b
        else:
            return [layer.delta_w], [layer.delta_b]

    def apply_layer_weights(self, layer, w, b):
        if isinstance(layer.weights, list):
            layer.weights, layer.biases = w[:], b[:]
        else:
            layer.weights, layer.biases = w[0].copy(), b[0].copy()

    def feedforward(self, in_back_propogation = True):
        a = self.layer_input_a
        for layer in self.layers:
            a = layer.feedforward(a, in_back_propogation)
        return a

    def cost(self, a_output):
        if self.cost_func:
            cost = self.cost_func.cost(a_output, y = 0)
        else:
            # suppose output to a single neuron with no activation function, quadratic cost
            cost = np.square(np.sum(a_output)) / 2
            #cost = -np.log(1-Sigmoid.f(np.sum(a_output)))
        return cost

    def deriv_a(self, a_output):
        if self.cost_func:
            y = 0
            return y
        else:
            # suppose output to a single neuron with no activation function, quadratic cost
            return np.ones_like(a_output) * np.sum(a_output)
            # return np.ones_like(a_output) * Sigmoid.f(np.sum(a_output))

    def test(self, ut):
        a = self.feedforward(in_back_propogation = True)
        deriv_a = self.deriv_a(a)
        deriv_w, deriv_b = {}, {}
        for layer in self.layers[::-1]:
            deriv_a = layer.back_propogation(deriv_a)
            if layer.trainable:
                deriv_w[layer], deriv_b[layer] = self.get_layer_deriv_weights(layer)
        for layer in self.layers:
            if layer.trainable:
                w, b = self.get_layer_weights(layer)
                if not isinstance(w, list): w = [w]
                if not isinstance(b, list): b = [b]
                for j in range(len(w)):
                    for i in range(w[j].size):
                        approx_deriv_wi = self.get_approv_deriv_w(layer, w, b, j, i)
                        deriv_wi = np.ravel(deriv_w[layer][j])[i]
                        Debug.print_('layer: {}, index: {}, deriv_wi: {}, approx_deriv_wi: {}'.format(layer.__class__.__name__, i, deriv_wi, approx_deriv_wi))
                        ut.assertAlmostEqual(deriv_wi, approx_deriv_wi, delta = 0.001*abs(deriv_wi))
                for j in range(len(b)):
                    for i in range(b[j].size):
                        approx_deriv_bi = self.get_approv_deriv_b(layer, w, b, j, i)
                        deriv_bi = np.ravel(deriv_b[layer][j])[i]
                        Debug.print_('layer: {}, index: {}, deriv_bi: {}, approx_deriv_bi: {}'.format(layer.__class__.__name__, i, deriv_bi, approx_deriv_bi))
                        ut.assertAlmostEqual(deriv_bi, approx_deriv_bi, delta = 0.001*abs(deriv_bi))

    def get_approv_deriv_w(self, layer, w, b, w_list_index, w_index):
        wi = np.ravel(w[w_list_index])[w_index]
        delta_wi = wi / 100
        np.ravel(w[w_list_index])[w_index] = wi + delta_wi
        self.apply_layer_weights(layer, w, b)
        a_output = self.feedforward()
        cost1 = self.cost(a_output)
        np.ravel(w[w_list_index])[w_index] = wi - delta_wi
        self.apply_layer_weights(layer, w, b)
        a_output = self.feedforward()
        cost2 = self.cost(a_output)
        approx_deriv_wi = (cost1 - cost2) / (2*delta_wi)
        np.ravel(w[w_list_index])[w_index] = wi
        self.apply_layer_weights(layer, w, b)
        return approx_deriv_wi

    def get_approv_deriv_b(self, layer, w, b, b_list_index, b_index):
        wi = np.ravel(b[b_list_index])[b_index]
        delta_wi = max(wi / 100, 0.0001)
        np.ravel(b[b_list_index])[b_index] = wi + delta_wi
        self.apply_layer_weights(layer, w, b)
        a_output = self.feedforward()
        cost1 = self.cost(a_output)
        np.ravel(b[b_list_index])[b_index] = wi - delta_wi
        self.apply_layer_weights(layer, w, b)
        a_output = self.feedforward()
        cost2 = self.cost(a_output)
        approx_deriv_wi = (cost1 - cost2) / (2*delta_wi)
        np.ravel(b[b_list_index])[b_index] = wi
        self.apply_layer_weights(layer, w, b)
        return approx_deriv_wi


class TestConvLayer(unittest.TestCase):
    def setUp(self):
        #self.conv_layer = ConvLayer([3,3,2,2])
        self.conv_layer = ConvLayer([3,3,1,1])

    def tearDown(self):
        pass

    def x_test_conv2d(self):
        # a: 2*(6*6), f: 2*2*(3*3)
        a = np.arange(36*2).reshape(2, 6, 6)
        f = np.array([[0,0,1,-1,0,0,0,0,1],
                      [0,0,1,-1,0,0,0,0,1],
                      [1,0,0,0,0,-1,1,0,0],
                      [1,0,0,0,0,-1,1,0,0]]).reshape(2, 2, 3, 3)
        # z: 2*(4*4)
        z = self.conv_layer.conv2d(a, f)
        z1 = np.array([10,11,12,13,16,17,18,19,22,23,24,25,28,29,30,31])*2+36
        z2 = np.array([4,5,6,7,10,11,12,13,16,17,18,19,22,23,24,25])*2+36
        z_expect = np.concatenate((z1,z2)).reshape(2, 4, 4)
        Debug.print_('test_conv2d:', 'z:', z, 'z_expect:', z_expect)
        self.assertTrue((z == z_expect).all())

    def x_test_back_conv2d(self):
        # a: 2*(6*6), f: 2*(4*4)
        a = np.arange(36*2).reshape(2, 6, 6)
        f = np.array([[0,0,0,1,-1,0,0,0,0,0,0,0,1,0,0,0],
                      [1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1]]).reshape(2, 4, 4)
        # z: 2*2*(3*3)
        z = self.conv_layer.back_conv2d(a, f)
        z1 = np.array([15,16,17,21,22,23,27,28,29])
        z2 = np.array([12,13,14,18,19,20,24,25,26])
        z_expect = np.concatenate((z1, z1+36, z2, z2+36)).reshape(2, 2, 3, 3)
        Debug.print_('test_back_conv2d:', 'z:', z, 'z_expect:', z_expect)
        self.assertTrue((z == z_expect).all())

    def x_test_feedforward(self):
        pass

    def x_test_back_propogation(self):
        def back_with_weights(x, tiny_delta_weights, weight_index):
            self.conv_layer = ConvLayer([3,3,1,1])
            self.conv_layer.weights = np.array([[ 0.08305444, -0.16140855,  0.08469047],
                                                [ 1.21509352,  0.47382514, -0.78398948],
                                                [ 0.93826142,  1.3423708 ,  1.84067832]]).reshape(self.conv_layer.weights.shape)
            np.ravel(self.conv_layer.weights)[weight_index] += tiny_delta_weights
            Debug.print_('test_back_propogation:', 'weight index:', weight_index, 'weights:', self.conv_layer.weights)
            #self.conv_layer.biases = origin_weights + tiny_delta_weights
            z = self.conv_layer.feedforward(a, in_back_propogation = True)
            # suppose one neuron in the next layer, with weights all 0.1, bias is zero, sigmoid and cross-entropy cost, y is zero
            y_ = Sigmoid.f(z.sum()-40)
            C = -np.log(1 - y_)  # suppose y = 0
            delta = y_
            d = self.conv_layer.back_propogation(delta * np.ones_like(z))
            return C, np.ravel(self.conv_layer.delta_w)[weight_index].copy()
        tiny_delta_weights = 0.0001
        #a = np.random.rand(1, 6, 6)
        a = np.array([0.74341247,  0.47463755,  0.00992929,  0.95730784,  0.20542349,
                        0.24606582,  0.5627104 ,  0.18438329,  0.89370057,  0.73840308,
                        0.96136674,  0.19538822,  0.1619067 ,  0.02462808,  0.85983933,
                        0.92236065,  0.87674389,  0.68733282,  0.4138197 ,  0.41656749,
                        0.38043692,  0.78814061,  0.30552122,  0.44576086,  0.79040761,
                        0.78019093,  0.95638804,  0.2221817 ,  0.18427876,  0.53748266,
                        0.23379542,  0.27326781,  0.14063543,  0.24078563,  0.54046106,
                        0.09593265]).reshape(1, 6, 6);
        cost1, delta_w1 = back_with_weights(a, tiny_delta_weights, weight_index = 1)
        cost2, delta_w2 = back_with_weights(a, -tiny_delta_weights, weight_index = 1)
        delta_w = (delta_w1 + delta_w2) /2
        gradient_w = (cost1 - cost2) / 2 / tiny_delta_weights
        Debug.print_('test_back_propogation:', 'delta_w1:', delta_w1, 'delta_w2:', delta_w2)
        Debug.print_('cost1:', cost1, 'cost2:', cost2)
        Debug.print_('gradient_w', gradient_w, 'delta_w:', delta_w)
        self.assertTrue(1)

    def xx_test_back_propogation(self):
        self.conv_layer = ConvLayer([3,3,2,2])
        a_input = np.array([0.74341247,  0.47463755,  0.00992929,  0.95730784,  0.20542349,
                        0.24606582,  0.5627104 ,  0.18438329,  0.89370057,  0.73840308,
                        0.96136674,  0.19538822,  0.1619067 ,  0.02462808,  0.85983933,
                        0.92236065,  0.87674389,  0.68733282,  0.4138197 ,  0.41656749,
                        0.38043692,  0.78814061,  0.30552122,  0.44576086,  0.79040761,
                        0.78019093,  0.95638804,  0.2221817 ,  0.18427876,  0.53748266,
                        0.23379542,  0.27326781,  0.14063543,  0.24078563,  0.54046106,
                        0.09593265]).reshape(1, 6, 6);
        a_input = np.concatenate((a_input, 1-a_input)).reshape(2, 6, 6)
        test_back_prop = TestBackProp(self.conv_layer, a_input)
        test_back_prop.test(self)

    def xx_test_back_propogation(self):
        self.conv_layer = ConvLayer([3,3,2,2])
        #self.pooling_layer = PoolingLayer()
        #self.fc_layer = FcLayer([8, 30, 1])
        self.fc_layer = FcLayer([8*4, 30, 1])
        a_input = np.array([0.74341247,  0.47463755,  0.00992929,  0.95730784,  0.20542349,
                        0.24606582,  0.5627104 ,  0.18438329,  0.89370057,  0.73840308,
                        0.96136674,  0.19538822,  0.1619067 ,  0.02462808,  0.85983933,
                        0.92236065,  0.87674389,  0.68733282,  0.4138197 ,  0.41656749,
                        0.38043692,  0.78814061,  0.30552122,  0.44576086,  0.79040761,
                        0.78019093,  0.95638804,  0.2221817 ,  0.18427876,  0.53748266,
                        0.23379542,  0.27326781,  0.14063543,  0.24078563,  0.54046106,
                        0.09593265]).reshape(1, 6, 6);
        a_input = np.concatenate((a_input, 1-a_input)).reshape(2, 6, 6)
        #test_back_prop = TestBackProp([self.conv_layer, self.pooling_layer, self.fc_layer], a_input)
        test_back_prop = TestBackProp([self.conv_layer, self.fc_layer], a_input)
        test_back_prop.test(self)

    def xx_test_back_propogation(self):
        #self.conv_layer = ConvLayer([3,3,2,2])
        #self.pooling_layer = PoolingLayer()
        #self.fc_layer = FcLayer([8, 30, 1])
        self.fc_layer = FcLayer([72, 30, 1])
        a_input = np.array([0.74341247,  0.47463755,  0.00992929,  0.95730784,  0.20542349,
                        0.24606582,  0.5627104 ,  0.18438329,  0.89370057,  0.73840308,
                        0.96136674,  0.19538822,  0.1619067 ,  0.02462808,  0.85983933,
                        0.92236065,  0.87674389,  0.68733282,  0.4138197 ,  0.41656749,
                        0.38043692,  0.78814061,  0.30552122,  0.44576086,  0.79040761,
                        0.78019093,  0.95638804,  0.2221817 ,  0.18427876,  0.53748266,
                        0.23379542,  0.27326781,  0.14063543,  0.24078563,  0.54046106,
                        0.09593265]).reshape(1, 6, 6);
        a_input = np.concatenate((a_input, 1-a_input)).reshape(2, 6, 6)
        #test_back_prop = TestBackProp([self.conv_layer, self.pooling_layer, self.fc_layer], a_input)
        test_back_prop = TestBackProp([self.fc_layer], a_input)
        test_back_prop.test(self)


if __name__ == '__main__':
    Debug.ENABLE = True
    unittest.main()