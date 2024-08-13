import numpy as np

class Dense():
    def __init__(self, inputs, outputs) -> None:
        self._input_len = inputs
        self._output_len = outputs
        self._weights = np.random.randn(outputs, inputs)
        self._last_input = np.array([[0.0], [0.0]], dtype=np.float32)
        self._bias = np.random.randn(outputs, 1)

    def feedforward(self, x):
        self._last_input = x
        return np.dot(self._weights, x)+self._bias

    def learn(self, lr, xi):
        dW = np.dot(xi, self._last_input.T)
        dB = np.sum(xi, axis=1, keepdims=True)
        self._weights -= lr*dW
        self._bias -= lr*dB
        return np.dot(self._weights.T, xi)


class Activation():

    leaky_relu_coef = 0.0

    def __init__(self, activation_fn, activation_fn_der) -> None:
        self._act_fn = activation_fn
        self._act_fn_der = activation_fn_der
        self._last_input = 0

    def feedforward(self, x):
        self._last_input = x
        res = np.zeros(x.shape, dtype=np.float32)
        for i, v in enumerate(x):
            res[i] = self._act_fn(v)
        return res

    def learn(self, xi):
        return xi*self._act_fn_der(self._last_input)

    def sigmoid(x) -> np.float32:
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_derivative(x) -> np.float32:
        return Activation.sigmoid(x)*(1.0-Activation.sigmoid(x))

    def relu(x) -> np.float32:
        return max(0, x)
    
    def relu_derivative(x) -> np.float32:
        return 0 if x<=0 else 1

    def tanh(x) -> np.float32:
        return np.tanh(x)
    
    def tanh_derivative(x) -> np.float32:
        return 1.0-(np.tanh(x)**2)

    def leaky_relu(x) -> np.float32:
        return max(Activation.leaky_relu_coef*x, x)
    
    def leaky_relu_derivative(x) -> np.float32:
        return Activation.leaky_relu_coef if x <= 0 else 1

    def softmax(xs: np.ndarray):
        shiftx = xs - np.max(xs)
        return np.exp(shiftx) / np.sum(np.exp(shiftx))

    def softmax_derivative(xs):
        ## TODO:
        raise NotImplementedError


def loss(ys, y_real_s):
    return np.mean((ys - y_real_s) ** 2)

def loss_derivative(ys, y_real_s, num_of_input):
    return 2 * (ys - y_real_s) / num_of_input

def main():
    xor_x = np.array([[[0], [0]],[[0], [1]],[[1], [0]],[[1], [1]]], dtype=np.float32)
    xor_y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    l1 = Dense(2, 2)
    l2 = Activation(Activation.sigmoid, Activation.sigmoid_derivative)
    l3 = Dense(2, 1)
    l4 = Activation(Activation.sigmoid, Activation.sigmoid_derivative)

    for it in range(1, 10000):
        for i, train_sample in enumerate(xor_x):
            #predict
            result = l4.feedforward(l3.feedforward(l2.feedforward(l1.feedforward(train_sample))))
            l = loss(result, xor_y[i])
            print(it," iteration, loss: ", l)

            #train
            xi = np.array(loss_derivative(result, xor_y[i], 2)).reshape((1, 1))
            xi = l4.learn(xi)
            xi = l3.learn(0.5, xi)
            xi = l2.learn(xi)
            l1.learn(0.5, xi)

    for train_sample in xor_x:
            result = l4.feedforward(l3.feedforward(l2.feedforward(l1.feedforward(train_sample))))
            print(train_sample, " => ", result)


if __name__ == "__main__":
    main()