from enum import Enum
import numpy as np


def mse(predicted: np.ndarray, output: np.ndarray):
    return np.mean((predicted - output)**2)


def mse_derivative(predicted: np.ndarray, output: np.ndarray):
    return 2.0 * (predicted - output) / float(predicted.size)


def mae(predicted: np.ndarray, output: np.ndarray):
    return np.mean(np.abs(predicted - output))


def mae_derivative(predicted: np.ndarray, output: np.ndarray):
    return np.sign(predicted - output) / float(predicted.size)


def softmax(predicted: np.ndarray):
    exp_preds = np.exp(predicted - np.max(predicted, axis=1, keepdims=True))
    return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)


def cross_entropy(predicted: np.ndarray, output: np.ndarray):
    epsilon = 1e-15
    p = softmax(predicted)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    log_likelihood = -np.log(p[range(output.shape[0]), output.argmax(axis=1)])
    return np.sum(log_likelihood) / output.shape[0]

def cross_entropy_derivative(predicted: np.ndarray, output: np.ndarray):
    epsilon = 1e-15
    p = softmax(predicted)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    m = output.shape[0]
    grad = p
    grad[range(m), output.argmax(axis=1)] -= 1
    return grad / m



def log_loss(predicted: np.ndarray, output: np.ndarray):
    epsilon = 1e-15
    predicted = np.clip(predicted, epsilon, 1.0 - epsilon)
    return -np.mean(output * np.log(predicted) + (1 - output) * np.log(1 - predicted))


def log_loss_derivative(predicted: np.ndarray, output: np.ndarray):
    epsilon = 1e-15
    predicted = np.clip(predicted, epsilon, 1.0 - epsilon)
    return (predicted - output) / (predicted * (1 - predicted) * predicted.size)


class Loss:

    class LossType(Enum):
        MSE=1,
        MAE = 2,
        CROSS_ENTROPY = 3,
        LOG_LOSS = 4,

    def __init__(self, loss_type: LossType) -> None:
         match loss_type:
            case Loss.LossType.MSE:
                self._loss_fn = mse
                self._loss_fn_der = mse_derivative
            case Loss.LossType.MAE:
                self._loss_fn = mae
                self._loss_fn_der = mae_derivative
            case Loss.LossType.CROSS_ENTROPY:
                self._loss_fn = cross_entropy
                self._loss_fn_der = cross_entropy_derivative
            case Loss.LossType.LOG_LOSS:
                self._loss_fn = log_loss
                self._loss_fn_der = log_loss_derivative
            case _:
                raise Exception("Unexpected loss function type: ", loss_type)
             
    def get_loss(self, predicted, output):
        return self._loss_fn(predicted, output)
    
    def get_loss_derivative(self, predicted, output):
        return self._loss_fn_der(predicted, output)
