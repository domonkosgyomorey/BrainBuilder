import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class ILayer(ABC):

    @abstractmethod
    def feedforward(self, xs):
        pass

    @abstractmethod
    def learn(self, lr, xi):
        pass

class Dense(ILayer):
    def __init__(self, inputs, outputs) -> None:
        self._input_len = inputs
        self._output_len = outputs
        self._weights = np.random.randn(outputs, inputs) * np.sqrt(2. / inputs)
        self._last_input = np.array([[0.0], [0.0]], dtype=np.float32)
        self._bias = np.random.randn(outputs, 1)

    def feedforward(self, x):
        self._last_input = x
        return np.dot(self._weights, x)+self._bias

    def learn(self, lr, xi):
        dw = np.dot(xi, self._last_input.T)
        db = np.sum(xi, axis=1, keepdims=True)
        self._weights -= lr*dw
        self._bias -= lr*db
        return np.dot(self._weights.T, xi)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)



def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0-(np.tanh(x)**2)


def leaky_relu(x):
    return [[max(Activation.leaky_relu_coef * c, c)] for c in x]

def leaky_relu_derivative(x):
    return [[Activation.leaky_relu_coef] if c <= 0 else [1] for c in x]


class Activation(ILayer):

    leaky_relu_coef = 0.1

    class ActivationType(Enum):
        ReLU = 1,
        Sigmoid = 2,
        TanH = 3,
        LeakyReLU = 4,

    def __init__(self, activation_type: ActivationType) -> None:
        match activation_type:
            case Activation.ActivationType.ReLU:
                self._act_fn = lambda x: relu(x)
                self._act_fn_der = lambda x: relu_derivative(x)
            case Activation.ActivationType.Sigmoid:
                self._act_fn = lambda x: sigmoid(x)
                self._act_fn_der = lambda x: sigmoid_derivative(x)
            case Activation.ActivationType.TanH:
                self._act_fn = lambda x: tanh(x)
                self._act_fn_der = lambda x: tanh_derivative(x)
            case Activation.ActivationType.LeakyReLU:
                self._act_fn = lambda x: leaky_relu(x)
                self._act_fn_der = lambda x: leaky_relu_derivative(x)
            case _:
                raise Exception("Unexpected activation type: ", activation_type)

        self._last_input = 0

    def feedforward(self, xs):
        self._last_input = xs
        res = np.zeros(xs.shape, dtype=np.float32)
        for i, v in enumerate(xs):
            res[i] = np.array(self._act_fn(v))
        return res

    def learn(self, lr, xi):
        return xi*self._act_fn_der(self._last_input)