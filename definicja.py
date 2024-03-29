# -*- coding: utf-8 -*-
"""lab3/4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pNaXYDthIUrFxdwtDCTxwndUbLRb9kS2
"""

import numpy as np
np. random.seed(100)



class activation_fcn(object):
    def __init__(self):
        pass

    def output(self, layer, name, derivative=False):
        method = getattr(self, str(name),"Invalid_function")
        return method(layer, derivative)

    def invalid_function(self, *args):
        print("Error: Invalid activation function")
        return None

    def linear(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = layer['activation_potential']
        else:
            out = np.ones(shape=np.shape(layer['activation_potential']))
        return out

    def tanh(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = np.tanh(layer['activation_potential'])
        else:
            out = 1.0 - np.tanh(layer['activation_potential']) ** 2
        return out

    def relu(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = np.maximum(0, layer['activation_potential'])
        else:
            out = np.where(layer['activation_potential'] > 0, 1, 0)
        return out

    def logistic(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = 1.0 / (1.0 + np.exp(-layer['activation_potential']))
        else:
            out = layer['output'] * (1 - layer['output'])
        return out

"""Zad 2"""

class loss_fcn(object):
    def __init__(self):
        pass

    def loss(self, loss, expected, outputs, derivative):
        method = getattr(self, str(loss), lambda: "Invalid loss function")
        return method(expected, outputs, derivative)

    def invalid_function(self, *arg):
        print("Error: Invalid loss function")
        return None

    def mse(self, expected, outputs, derivative=False):
        error_value = 0
        if not derivative:
            error_value = 0.5 * np.power(expected - outputs, 2)
        else:
            error_value = -(expected - outputs)
        return error_value

    def binary_crossentropy(self, expected, outputs, derivative=False):
        outputs = np.clip(outputs, 1e-7, 1 - 1e-7)
        value = (expected * np.log(outputs) + (1 - expected) * np.log(1 - outputs))
        if not derivative:
            value = -np.mean(value)
        return value

"""Zad 3/4"""

#zad3
class Neural_network(object):
    def __init__(self):
        pass

    def __get_weights(self, args, init_weight='rand'):
        if init_weight == 'zero':
            return np.zeros(args)
        if init_weight == 'one':
            return np.ones(args)
        if init_weight == 'rand':
            return np.random.randn(*args)

        raise ValueError("Invalid value for init_weight parameter")

    def create_network(self, structure, init_weight='rand'):
        activation_function=activation_fcn
        self.nnetwork = [structure[0]]
        for i in range(1, len(structure)):

            weights = self.__get_weights((structure[i]['units'], structure[i-1]['units'] + structure[i]['bias']), init_weight)

            new_layer = {
                'weights': weights,
                'bias': structure[i]['bias'],
                'activation_function': structure[i]['activation_function'],
                'activation_potential': None,
                'delta': None,
                'output': None}
            self.nnetwork.append(new_layer)
        return self.nnetwork

    def train(self, nnetwork, x_train, y_train, l_rate=0.01, n_epoch=100, loss_function='mse', verbose=1):
        for epoch in range(n_epoch):
            sum_error = 0
            for iter, (x_row, y_row) in enumerate(zip(x_train, y_train)):

                self.forward_propagate(nnetwork, x_row)

                loss = loss_fcn()
                sum_error = np.sum(loss.loss(loss_function, y_row, nnetwork[-1]['output'], derivative=False))

                self.backward_propagate(loss_function, nnetwork, y_row)

                self.update_weights(nnetwork, x_row, l_rate)

            if verbose > 0:
                print('>epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        print('Results: epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        return self.nnetwork

    def forward_propagate(self, nnetwork, inputs):
        for i in range(1, len(nnetwork)):
            layer = nnetwork[i]


            if layer['bias']:
                inputs = np.append(inputs, 1)

            activation_potential = np.dot(layer['weights'], inputs)
            layer['activation_potential'] = activation_potential

            layer['output'] = activation_fcn().output(layer, layer['activation_function'], derivative=False)

            inputs = layer['output']

        return nnetwork[-1]['output']

    def predict(self, nnetwork, inputs):
        predictions = []
        for sample in inputs:
            output = self.forward_propagate(nnetwork, sample)
            predictions.append(output)
        return predictions

    def backward_propagate(self, loss_function, nnetwork, expected):
        pass


    # Update network weights with error
    def update_weights(self, nnetwork, inputs, l_rate):
        pass

#zad5.1
structure = [{'type': 'input', 'units': 1},
            {'type': 'dense', 'units': 2, 'activation_function': 'linear', 'bias': True},
            {'type': 'dense', 'units': 1, 'activation_function': 'linear', 'bias': True}]

model = Neural_network()
network = model.create_network(structure, "zero")

n = 1
X = np.linspace(-5, 5, n).reshape(-1, 1)

predicted = model.predict(network, X)

print(predicted)

#zad5.2
structure = [{'type': 'input', 'units': 1},
            {'type': 'dense', 'units': 2, 'activation_function': 'tanh', 'bias': True},
            {'type': 'dense', 'units': 2, 'activation_function': 'tanh', 'bias': True},
            {'type': 'dense', 'units': 1, 'activation_function': 'tanh', 'bias': True}]

network = model.create_network(structure, "one")

n=1
X = np.linspace(-5, 5, n).reshape(-1, 1)

predicted = model.predict(network, X)
print(predicted)

#zad5.3
structure = [{'type': 'input', 'units': 1},
            {'type': 'dense', 'units': 8, 'activation_function': 'tanh', 'bias': True},
            {'type': 'dense', 'units': 8, 'activation_function': 'relu', 'bias': True},
            {'type': 'dense', 'units': 1, 'activation_function': 'linear', 'bias': True}]

network = model.create_network(structure, "rand")

n=1
X = np.linspace(-5, 5, n).reshape(-1, 1)

predicted = model.predict(network, X)
print(predicted)

def test_regression():
    # Read data
    X, Y = generate_regression_data()

    # Create network
    model = Neural_network()
    structure = [{'type': 'input', 'units': 1},
                 {'type': 'dense', 'units': 8, 'activation_function': 'tanh', 'bias': True},
                 {'type': 'dense', 'units': 8, 'activation_function': 'tanh', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'linear', 'bias': True}]

    network = model.create_network(structure)

    model.train(network, X, Y, 0.01, 4000, 'mse', 0)

    predicted = model.predict(network, X)
    std = np.std(predicted - Y)
    print("\nStandard deviation = {}".format(std))

    X_test = np.linspace(-7, 7, 100).reshape(-1, 1)
    X_test = np.array(X_test).tolist()
    predicted = model.predict(network, X_test)

    plt.plot(X, Y, 'r--o', label="Training data")
    plt.plot(X_test, predicted, 'b--x', label="Predicted")
    plt.legend()
    plt.grid()
    plt.show()


def generate_classification_data(n=30):
    # Class 1 - samples generation
    X1_1 = 1 + 4 * np.random.rand(n, 1)
    X1_2 = 1 + 4 * np.random.rand(n, 1)
    class1 = np.concatenate((X1_1, X1_2), axis=1)
    Y1 = np.ones(n)

    # Class 0 - samples generation
    X0_1 = 3 + 4 * np.random.rand(n, 1)
    X0_2 = 3 + 4 * np.random.rand(n, 1)
    class0 = np.concatenate((X0_1, X0_2), axis=1)
    Y0 = np.zeros(n)

    X = np.concatenate((class1, class0))
    Y = np.concatenate((Y1, Y0))

    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]

    return X, Y, idx0, idx1


def test_classification():
    # Read data
    X, Y, idx0, idx1 = generate_classification_data()

    # Create network
    model = Neural_network()
    structure = [{'type': 'input', 'units': 2},
                 {'type': 'dense', 'units': 4, 'activation_function': 'relu', 'bias': True},
                 {'type': 'dense', 'units': 4, 'activation_function': 'relu', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'logistic', 'bias': True}]

    network = model.create_network(structure)

    model.train(network, X, Y, 0.0001, 2000, 'binary_cross_entropy', 0)

    y = model.predict(network, X)
    t = 0
    for n, m in zip(y, Y):
        t += 1 - np.abs(np.round(np.array(n)) - np.array(m))
        print(f"pred = {n}, pred_round = {np.round(n)}, true = {m}")

    ACC = t / len(X)
    print(f"\nClassification accuracy = {ACC * 100}%")

    # Plotting decision regions
    xx, yy = np.meshgrid(np.arange(0, 8, 0.1),
                         np.arange(0, 8, 0.1))

    X_vis = np.c_[xx.ravel(), yy.ravel()]

    h = model.predict(network, X_vis)
    h = np.array(h) >= 0.5
    h = np.reshape(h, (len(xx), len(yy)))

    plt.contourf(xx, yy, h, cmap='jet')
    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", edgecolors="white", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", edgecolors="white", label="class 0")
    plt.show()


generate_classification_data(30)
test_classification()

generate_regression_data(30)
test_regression()