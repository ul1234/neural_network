#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time, json
from debug import Debug
from neuron import *
from layer import *


class Network(object):
    def __init__(self, layers, cost, optimizer = Sgd(0.1), regularization = RegularNone()):
        self.layers = layers
        last_layer = layers[-1]
        last_layer.set_cost_to_last_layer(cost)
        self.cost_func = cost
        self.optimizer = optimizer
        self.regularization = regularization
        self.stop = EarlyStop(10)

    def network_info(self):
        layers_name = [layer.__class__.__name__ for layer in self.layers]
        return 'Layers: {}'.format(layers_name)

    def set_stop(self, stop = EarlyStop(10)):
        self.stop = stop

    def feedforward(self, data, in_back_propogation = False):
        for layer in self.layers:
            data = layer.feedforward(data, in_back_propogation)
        return data

    def back_propogation(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.back_propogation(delta)
        return delta

    def update_weights(self, training_size):
        for layer in self.layers:
            layer.update_weights(training_size, self.optimizer, self.regularization)

    @classmethod
    def unpack_data(cls, packed_data):
        data_x = np.array([x for x, y in packed_data])
        data_y = np.array([y for x, y in packed_data])
        return (data_x, data_y)
        
    @classmethod
    def pack_data(cls, input_data, label_data):
        return list(zip(input_data, label_data))

    def train_batch(self, batch_data, training_size):
        # batch_data: [(x, y), (x, y) ... ]
        # batch_input_data: shape (num_batches, data_shape)
        # batch_label_data: shape (num_batches, data_shape)
        batch_input_data, batch_label_data = self.unpack_data(batch_data)
        self.feedforward(batch_input_data, in_back_propogation = True)
        self.back_propogation(batch_label_data)
        self.update_weights(training_size)

    def train(self, training_data, mini_batch_size, test_data = []):
        def print_training_info(test_data_accuracy = None):
            if not hasattr(print_training_info, 'training_epoch'): print_training_info.training_epoch = -1
            print_training_info.training_epoch += 1
            training_data_accuracy, training_data_cost = self.accuracy(training_data)
            if test_data_accuracy:
                print('epoch %d: cost %.3f training accuracy %.2f%%, test accuracy %.2f%%, elapsed: %.1fs' \
                    % (print_training_info.training_epoch, training_data_cost, 100*training_data_accuracy, 100*test_data_accuracy, time.time() - time_start))
            else:
                print('epoch %d: cost %.3f training accuracy %.2f%%, elapsed: %.1fs' \
                    % (print_training_info.training_epoch, training_data_cost, 100*training_data_accuracy, time.time() - time_start))
        print(self.network_info())
        # training_data, [(x0, y0), (x1, y1), ...]
        training_size = len(training_data)
        time_start = time.time()
        test_data_accuracy = self.accuracy(test_data)[0] if test_data else None
        print_training_info(test_data_accuracy)
        while not self.stop.stop(test_data_accuracy):
            np.random.shuffle(training_data)
            start = 0
            while start < training_size:
                batch_data = training_data[start:min(start+mini_batch_size, training_size)]
                start += mini_batch_size
                self.train_batch(batch_data, training_size)
            test_data_accuracy = self.accuracy(test_data)[0] if test_data else None
            print_training_info(test_data_accuracy)

    def accuracy(self, data_in):
        input_data, label_data = self.unpack_data(data_in)
        data_out = self.feedforward(input_data)
        cost = self.cost_func.cost(data_out, label_data)
        num_pass = (np.argmax(data_out, axis = 1) == np.argmax(label_data, axis = 1)).sum()
        accuracy = num_pass * 1.0 / len(data_in)
        return (accuracy, cost)

    def get_layers_data(self):
        layers_data = {}
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                layer_data = {'layer': layer.__class__.__name__,
                              'weights_shape': layer.weights_shape,
                              'biases_shape': layer.biases_shape,
                              'weights': layer.weights.tolist(),
                              'biases': layer.biases.tolist(),
                              'activation': layer.activation.__name__}
            else:
                layer_data = {'layer': layer.__class__.__name__}
            layers_data['Layer%d'%i] = layer_data
        return layers_data

    def apply_layers_data(self, layers_data):
        for i in range(len(layers_data)):
            layer_data = layers_data['Layer%d'%i]
            layer = self.layers[i]
            assert layer.__class__.__name__ == layer_data['layer']
            if layer.trainable:
                assert layer.weights_shape == tuple(layer_data['weights_shape'])
                assert layer.biases_shape == tuple(layer_data['biases_shape'])
                layer.weights = np.array(layer_data['weights'])
                layer.biases = np.array(layer_data['biases'])

    def save(self, filename = 'layers_data.txt'):
        layers_data = self.get_layers_data()
        with open(filename, 'w') as f:
            json.dump(layers_data, f)

    def load(self, filename = 'layers_data.txt'):
        layers_data = json.load(open(filename, 'r'))
        self.apply_layers_data(layers_data)


if __name__ == '__main__':
    from data_load import DataLoad
    data_load = DataLoad()

    option = 4

    if option == 1:     # all full connected network
        # 50000, 10000, 10000
        training_data, validation_data, test_data = data_load.load_mnist()
        fc_layer1 = FullConnectedLayer((28*28, 30))
        fc_layer2 = FullConnectedLayer((30, 10))
        net = Network([fc_layer1, fc_layer2],
                      cost = CrossEntropy(Sigmoid),
                      optimizer = Sgd(0.1),
                      regularization = RegularL2(0.1))
        net.train(training_data, 30, test_data = test_data)
    elif option == 2:   # CNN
        training_data, validation_data, test_data = data_load.load_mnist(shape = ((1,28,28), -1))
        #training_data = training_data[:1000]
        #test_data = test_data[:1000]
        conv_layer1 = ConvLayer((16, 1, 3, 3))  # output 26*26*16
        pooling_layer1 = PoolingLayer() # output 13*13*16
        fc_layer1 = FullConnectedLayer((13*13*16, 30))
        fc_layer2 = FullConnectedLayer((30, 10))
        net = Network([conv_layer1, pooling_layer1, fc_layer1, fc_layer2],
                      cost = CrossEntropy(Sigmoid),
                      optimizer = Sgd(0.1),
                      regularization = RegularL2(0.1))
        net.train(training_data, 30, test_data = test_data)
    elif option == 3:   # deeper CNN
        training_data, validation_data, test_data = data_load.load_mnist(shape = ((1,28,28), -1))
        training_data = training_data[:1000]
        test_data = test_data[:1000]
        conv_layer1 = ConvLayer((16, 1, 5, 5))  # output 24*24*16
        pooling_layer1 = PoolingLayer() # output 12*12*16
        conv_layer2 = ConvLayer((32, 16, 3, 3))  # output 10*10*32
        pooling_layer2 = PoolingLayer() # output 5*5*32
        fc_layer1 = FullConnectedLayer((5*5*32, 100))
        fc_layer2 = FullConnectedLayer((100, 10))
        net = Network([conv_layer1, pooling_layer1, conv_layer2, pooling_layer2, fc_layer1, fc_layer2],
                      cost = CrossEntropy(Sigmoid),
                      optimizer = Sgd(0.1),
                      regularization = RegularL2(0.1))
        net.train(training_data, 30, test_data = test_data)
    elif option == 4:   # RNN
        training_data, validation_data, test_data = data_load.load_mnist(shape = ((28,28), -1))
        #training_data = training_data[:5000]
        #test_data = test_data[:5000]
        only_RNN_layer = True
        if only_RNN_layer:
            rnn_layer1 = RecurrentLayer(28, 200, 10)
            net = Network([rnn_layer1],
                          cost = CrossEntropy(Sigmoid),
                          optimizer = Sgd(0.01))
        else:
            rnn_layer1 = RecurrentLayer(28, 100, 100)
            fc_layer1 = FullConnectedLayer((100, 10))
            net = Network([rnn_layer1, fc_layer1],
                           cost = CrossEntropy(Sigmoid),
                           optimizer = Sgd(0.01))
        net.train(training_data, 30, test_data = test_data)

    #net.save('layers_data.txt')
    #net.load('layers_data.txt')
    #input('press any key to continue...')

