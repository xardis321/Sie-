import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

class activation_fcn(object):
    def __init__(self):
        pass

    def output(self, layer, name, derivative=False):
        # Get the specified activation function or print an error for an invalid function
        method = getattr(self, str(name), "Invalid_function")
        return method(layer, derivative)

    def Invalid_function(self, *arg):
        print("Error: Invalid activation function")
        return None

    def linear(self, layer, derivative=False):
        # Linear activation function or its derivative
        out = layer['activation_potential'] if not derivative else np.ones(shape=np.shape(layer['activation_potential']))
        return out

    def logistic(self, layer, derivative=False):
        # Logistic (sigmoid) activation function or its derivative
        out = 1.0 / (1.0 + np.exp(-layer['activation_potential'])) if not derivative else layer['output'] * (1.0 - layer['output'])
        return out

    def tanh(self, layer, derivative=False):
        # Hyperbolic tangent (tanh) activation function or its derivative
        out = (np.exp(layer['activation_potential']) - np.exp(-layer['activation_potential'])) / (
            np.exp(layer['activation_potential']) + np.exp(-layer['activation_potential'])) + 1e-8 if not derivative else 1.0 - np.power(layer['output'], 2)
        return out

    def relu(self, layer, derivative=False):
        # Rectified Linear Unit (ReLU) activation function or its derivative
        out = np.maximum(0, layer['activation_potential']) if not derivative else layer['activation_potential'] >= 0
        return out

class loss_fcn(object):
    def __init__(self):
        pass

    def loss(self, loss, expected, outputs, derivative):
        # Get the specified loss function or print an error for an invalid function
        method = getattr(self, str(loss), lambda: "Invalid loss function")
        return method(expected, outputs, derivative)

    def Invalid_function(self, *arg):
        print("Error: Invalid loss function")
        return None

    def mse(self, expected, outputs, derivative=False):
        # Mean Squared Error (MSE) loss function or its derivative
        error_value = 0.5 * np.power(expected - outputs, 2) if not derivative else -(expected - outputs)
        return error_value

    def binary_cross_entropy(self, expected, outputs, derivative=False):
        # Binary Cross-Entropy loss function or its derivative
        error_value = -expected * np.log(outputs + 1e-8) - (1 - expected) * np.log(1 - outputs + 1e-8) if not derivative else -(expected / outputs - (1 - expected + 1e-8) / (1 - outputs + 1e-8))
        return error_value

class Neural_network(object):
    def __init__(self):
        pass

    def create_network(self, structure):
        # Initialize the neural network based on the specified structure
        self.nnetwork = [structure[0]]
        for i in range(1, len(structure)):
            new_layer = {
                'weights': np.random.randn(structure[i]['units'], structure[i-1]['units'] + structure[i]['bias']),
                'bias': structure[i]['bias'],
                'activation_function': structure[i]['activation_function'],
                'activation_potential': None,
                'delta': None,
                'output': None,
                'prev_delta': np.zeros((structure[i]['units'], structure[i-1]['units'] + structure[i]['bias']))
            }
            self.nnetwork.append(new_layer)
        return self.nnetwork

    def forward_propagate(self, nnetwork, inputs):
        # Perform forward propagation through the neural network
        af = activation_fcn()
        inp = inputs.copy()
        for i in range(1, len(nnetwork)):
            if nnetwork[i]['bias'] == True:
                inp = np.append(inp, 1)
            nnetwork[i]['activation_potential'] = np.matmul(nnetwork[i]['weights'], inp).flatten()
            nnetwork[i]['output'] = af.output(nnetwork[i], nnetwork[i]['activation_function'], derivative=False)
            inp = nnetwork[i]['output']
        return inp

    def backward_propagate(self, loss_function, nnetwork, expected):
        # Perform backward propagation through the neural network
        af = activation_fcn()
        loss = loss_fcn()
        N = len(nnetwork) - 1
        for i in range(N, 0, -1):
            errors = []
            if i < N:
                weights = nnetwork[i + 1]['weights']
                if nnetwork[i + 1]['bias'] == True:
                    weights = weights[:, :-1]
                errors = np.matmul(nnetwork[i + 1]['delta'], weights)
            else:
                errors = loss.loss(loss_function, expected, nnetwork[-1]['output'], derivative=True)

            nnetwork[i]['delta'] = np.multiply(errors, af.output(nnetwork[i], nnetwork[i]['activation_function'],
                                                                   derivative=True))
### Add momentum method, controlling the influence of previous updates on current ones
    def update_weights(self, nnetwork, inputs, l_rate, momentum_alpha):
        inp = inputs
        for i in range(1, len(nnetwork)):
            if nnetwork[i]['bias'] == True:
                inp = np.append(inp, 1)

            gradient_update = l_rate * np.matmul(nnetwork[i]['delta'].reshape(-1, 1), inp.reshape(1, -1))

            if 'prev_update' not in nnetwork[i]:
                nnetwork[i]['prev_update'] = np.zeros_like(gradient_update)
            momentum_update = momentum_alpha * nnetwork[i]['prev_update']

            nnetwork[i]['weights'] -= gradient_update + momentum_update

            nnetwork[i]['prev_update'] = gradient_update + momentum_update

            inp = nnetwork[i]['output']
## Add the darken_moody method to the train function

    def train(self, nnetwork, x_train, y_train, l_rate=0.01, n_epoch=100, loss_function='mse', verbose=1,
              lrate_method=None, l_rate_start=None, momentum_alpha=0.9, tau=100):  # Add tau - a constant
        l_rate_orig = l_rate_start if l_rate_start is not None else l_rate
        for epoch in range(n_epoch):
            sum_error = 0
            for iter, (x_row, y_row) in enumerate(zip(x_train, y_train)):
                self.forward_propagate(nnetwork, x_row)
                loss = loss_fcn()
                sum_error = np.sum(
                    loss.loss(loss_function, y_row, nnetwork[-1]['output'], derivative=False))
                self.backward_propagate(loss_function, nnetwork, y_row)

                if lrate_method == 'darken_moody':
                    l_rate = l_rate_orig / ((1.0 + epoch + 1e-8) * tau)  # Modify l_rate using the formula

                self.update_weights(nnetwork, x_row, l_rate, momentum_alpha)

            if verbose > 0:
                print('>epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        print('Results: epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        return nnetwork

    def predict(self, nnetwork, inputs):
        out = []
        for input in inputs:
            out.append(self.forward_propagate(nnetwork, input))
        return out


def generate_regression_data(n=30):
    # Generate regression dataset
    X = np.linspace(-5, 5, n).reshape(-1, 1)
    y = np.sin(2 * X) + np.cos(X) + 5
    # Simulate noise
    data_noise = np.random.normal(0, 0.2, n).reshape(-1, 1)
    # Generate training data
    Y = y + data_noise

    return X.reshape(-1, 1), Y.reshape(-1, 1)


def test_regression():
    # Read data
    X, Y = generate_regression_data()

    # Create network
    model = Neural_network()
    structure = [{'type': 'input', 'units': 1},
                 {'type': 'dense', 'units': 32, 'activation_function': 'tanh', 'bias': True},
                 {'type': 'dense', 'units': 16, 'activation_function': 'relu', 'bias': True},
                 {'type': 'dense', 'units': 8, 'activation_function': 'relu', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'linear', 'bias': True}]

    network = model.create_network(structure)

    model.train(network, X, Y, 0.001, 5000, 'mse', 0, 'darken_moody', 0.1, 0.1, 100)

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

    model.train(network, X, Y, 0.001, 2000, 'binary_cross_entropy', 0, 'darken_moody', 0.1, 0.9, 100)

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


generate_classification_data(10)
test_classification()

generate_regression_data(30)
test_regression()
