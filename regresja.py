import numpy as np

def activation_potential(input, weights, bias):
    """
    Calculates the activation potential of a neuron.

    Parameters:
    - input: Input data for the neuron.
    - weights: Weights associated with the neuron.
    - bias: Boolean indicating whether bias is present.

    Returns:
    - The activation potential of the neuron.
    """
    if bias:
        potential = np.dot(input, weights[1:]) + weights[0]
    else:
        potential = np.dot(input, weights)

    return potential

input_size = 3
bias = True
input_data = [0.1, 0.2, 0.3]
np.random.seed(1)
weights = [np.random.randn() for _ in range(input_size + bias)]

potential = activation_potential(input_data, weights, bias)
print("Neuron Potential:", potential)

def linear_activation(input, derivative=False):
    """
    Implements the linear activation function.

    Parameters:
    - input: Input value.
    - derivative: Boolean indicating whether to compute the derivative.

    Returns:
    - Output of the linear activation function or its derivative.
    """
    return 1 if derivative else input

def sigmoid_activation(input, derivative=False):
    """
    Implements the sigmoid activation function.

    Parameters:
    - input: Input value.
    - derivative: Boolean indicating whether to compute the derivative.

    Returns:
    - Output of the sigmoid activation function or its derivative.
    """
    output = 1 / (1 + np.exp(-input))
    return output * (1 - output) if derivative else output

def tanh_activation(input, derivative=False):
    """
    Implements the hyperbolic tangent (tanh) activation function.

    Parameters:
    - input: Input value.
    - derivative: Boolean indicating whether to compute the derivative.

    Returns:
    - Output of the tanh activation function or its derivative.
    """
    output = np.tanh(input)
    return 1 - output**2 if derivative else output

def relu_activation(input, derivative=False):
    """
    Implements the rectified linear unit (ReLU) activation function.

    Parameters:
    - input: Input value.
    - derivative: Boolean indicating whether to compute the derivative.

    Returns:
    - Output of the ReLU activation function or its derivative.
    """
    return 1 if derivative else max(0, input)

input_value = 2.0

linear_output = linear_activation(input_value)
linear_derivative = linear_activation(input_value, derivative=True)

sigmoid_output = sigmoid_activation(input_value)
sigmoid_derivative = sigmoid_activation(input_value, derivative=True)

tanh_output = tanh_activation(input_value)
tanh_derivative = tanh_activation(input_value, derivative=True)

relu_output = relu_activation(input_value)
relu_derivative = relu_activation(input_value, derivative=True)

print("Linear Activation Function:")
print("Output:", linear_output)
print("Derivative:", linear_derivative)

print("\nSigmoid Activation Function:")
print("Output:", sigmoid_output)
print("Derivative:", sigmoid_derivative)

print("\nTanh Activation Function:")
print("Output:", tanh_output)
print("Derivative:", tanh_derivative)

print("\nReLU Activation Function:")
print("Output:", relu_output)
print("Derivative:", relu_derivative)
